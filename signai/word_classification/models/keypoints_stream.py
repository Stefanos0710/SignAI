"""Pose/Keypoint stream: 49 MediaPipe keypoints -> 384-d pose feature.

    Keypoints (T, J, 3)
    -> HeatmapGenerator (heatmap.py)  (per-landmark Gaussian heatmaps)
    -> build_pose_encoder             (SlowOnly-R50 3D CNN -> 512-d)
    -> PoseFeatureProjection          (512 -> 384)
    -> Pose Feature (batch, 384)

Works directly on the dataset that is actually on disk
(signai/word_classification/dataset/processed/*_data.npz: X is
(N, 32, 147) float32, already shoulder-centered + shoulder-distance-scaled,
see signai/preprocessing/train_data.py::center_keypoints/normalize_keypoints).
Keypoint normalization is NOT repeated here -- it already happened in
preprocessing.py, so this module takes normalized (T, J, 3) keypoints as-is.
Heatmap rendering happens on-the-fly here in TensorFlow instead of being
baked into the dataset, so nothing about preprocessing.py or the cached
npz files needs to change.

Landmark layout this module assumes (matches signai/preprocessing/train_data.py
POSE_LANDMARKS + 21 left-hand + 21 right-hand, no face -- 7 + 21 + 21 = 49
landmarks, 49 * 3 = 147 features):
    0=nose, 1=left shoulder, 2=right shoulder, 3=left elbow, 4=right elbow,
    5=left wrist, 6=right wrist, 7..27=left hand, 28..48=right hand
"""
import argparse

import numpy as np
import tensorflow as tf
import keras

from heatmap import HeatmapGenerator, HEATMAP_HEIGHT, HEATMAP_WIDTH, HEATMAP_SIGMA
from visualization import KeypointsVisualizer, concat_real_clips, load_random_real_clip, real_frame_indices

PROJECTION_DIM = 384
ENCODER_OUTPUT_DIM = 512


def reshape_keypoints(flat, num_landmarks: int | None = None):
    """(..., T, J*3) -> (..., T, J, 3). J is inferred from the last dim if not given.

    The on-disk dataset stores each clip as (32, 147) -- keypoints flattened per
    frame. This just restores the (J, 3) structure.
    """
    flat = tf.convert_to_tensor(flat, dtype=tf.float32)
    if num_landmarks is None:
        num_landmarks = flat.shape[-1] // 3
    # (..., T, J*3) -> (..., T, J, 3)
    new_shape = tf.concat([tf.shape(flat)[:-1], [num_landmarks, 3]], axis=0)
    return tf.reshape(flat, new_shape)


def smooth_hand_keypoints(keypoints, window: int = 9, polyorder: int = 2):
    """Extra Savitzky-Golay smoothing pass over the hand landmarks (indices
    7:49) only, on top of whatever preprocessing.py already applied to the
    whole clip. Hand landmarks jitter far more than pose/wrist landmarks
    (MediaPipe's hand tracker is noisier than its pose tracker), so a
    dedicated, stronger pass just for hands helps visually without touching
    the already-stable pose landmarks (0:7).

    keypoints: (T, J, 3), real (non-padding) frames only -- run this before
    re-padding, since smoothing across a real/zero boundary would drag real
    hand positions toward the zero padding.
    """
    from scipy.signal import savgol_filter

    keypoints = np.asarray(keypoints, dtype=np.float32).copy()
    window = min(window, keypoints.shape[0] - (1 - keypoints.shape[0] % 2))
    if window < polyorder + 2:
        return keypoints  # too few frames to smooth meaningfully
    keypoints[:, 7:49, :] = savgol_filter(keypoints[:, 7:49, :], window, polyorder, axis=0)
    return keypoints


def _bottleneck_block(x, width, temporal_kernel=1, stride=1, name=None):
    """Standard ResNet-50 bottleneck (1x1x1 -> 3x3x3 -> 1x1x1, 4x channel
    expansion), inflated to 3D. temporal_kernel=1 keeps the middle conv
    temporally degenerate (per-frame 2D) -- the SlowOnly design (Feichtenhofer
    et al., SlowFast) only spends real temporal convs (kernel=3) on the later,
    more semantic stages (res4/res5), leaving the early stages cheap.
    """
    expansion = 4
    out_channels = width * expansion
    in_channels = x.shape[-1]
    shortcut = x

    y = keras.layers.Conv3D(width, 1, strides=1, padding="same", name=f"{name}_conv1")(x)
    y = keras.layers.BatchNormalization(name=f"{name}_bn1")(y)
    y = keras.layers.ReLU(name=f"{name}_relu1")(y)

    y = keras.layers.Conv3D(width, (temporal_kernel, 3, 3), strides=stride, padding="same", name=f"{name}_conv2")(y)
    y = keras.layers.BatchNormalization(name=f"{name}_bn2")(y)
    y = keras.layers.ReLU(name=f"{name}_relu2")(y)

    y = keras.layers.Conv3D(out_channels, 1, strides=1, padding="same", name=f"{name}_conv3")(y)
    y = keras.layers.BatchNormalization(name=f"{name}_bn3")(y)

    if stride != 1 or in_channels != out_channels:
        shortcut = keras.layers.Conv3D(out_channels, 1, strides=stride, padding="same", name=f"{name}_proj")(x)
        shortcut = keras.layers.BatchNormalization(name=f"{name}_proj_bn")(shortcut)

    out = keras.layers.Add(name=f"{name}_add")([y, shortcut])
    return keras.layers.ReLU(name=f"{name}_relu3")(out)


def _resnet_stage(x, width, num_blocks, temporal_kernel, stride, name):
    """One SlowOnly-R50 stage: `num_blocks` bottleneck blocks, spatial downsample
    (if any) only in the first block -- matches standard ResNet stage layout."""
    x = _bottleneck_block(x, width, temporal_kernel, stride=stride, name=f"{name}_block1")
    for i in range(2, num_blocks + 1):
        x = _bottleneck_block(x, width, temporal_kernel, stride=1, name=f"{name}_block{i}")
    return x


def build_pose_encoder(num_landmarks: int, height: int = HEATMAP_HEIGHT, width: int = HEATMAP_WIDTH,
                        num_frames: int | None = None) -> keras.Model:
    """SlowOnly-R50 3D CNN pose encoder (PoseConv3D-style): a real ResNet-50
    bottleneck backbone -- stage depths [3, 4, 6, 3], matching R50 -- inflated to
    3D. Following the SlowOnly design (Feichtenhofer et al., SlowFast), the two
    early stages (res2/res3) use temporally-degenerate convs (kernel=1,
    effectively per-frame 2D) and the two later/semantic stages (res4/res5) use
    real spatiotemporal convs (kernel=3): cheap early on, temporally aware once
    the representation is more abstract.

    Input:  (T, height, width, num_landmarks) -- one channel per landmark heatmap.
    Output: (512,). R50's native GlobalAveragePooling3D output is 2048-d (512 *
    the bottleneck's 4x expansion); one Dense feature head maps that down to 512
    so the rest of the pipeline (PoseFeatureProjection etc.) keeps its 512-d
    contract regardless of backbone depth.

    num_landmarks is a parameter, not hardcoded, so the same builder works for
    any keypoint layout/count -- the input's channel dim is just num_landmarks.
    """
    inputs = keras.Input(shape=(num_frames, height, width, num_landmarks))  # (T, H, W, J)

    # Stem (conv1): R50's 7x7 stride-2 conv + 3x3 stride-2 maxpool, temporal
    # kernel=1 (degenerate) -- the slow pathway doesn't need temporal info yet.
    x = keras.layers.Conv3D(64, (1, 7, 7), strides=(1, 2, 2), padding="same", name="stem_conv")(inputs)
    x = keras.layers.BatchNormalization(name="stem_bn")(x)
    x = keras.layers.ReLU(name="stem_relu")(x)
    x = keras.layers.MaxPool3D((1, 3, 3), strides=(1, 2, 2), padding="same", name="stem_pool")(x)
    # -> (T, H/4, W/4, 64)

    x = _resnet_stage(x, width=64, num_blocks=3, temporal_kernel=1, stride=1, name="res2")
    # -> (T, H/4, W/4, 256)
    x = _resnet_stage(x, width=128, num_blocks=4, temporal_kernel=1, stride=(1, 2, 2), name="res3")
    # -> (T, H/8, W/8, 512)
    x = _resnet_stage(x, width=256, num_blocks=6, temporal_kernel=3, stride=(1, 2, 2), name="res4")
    # -> (T, H/16, W/16, 1024)
    x = _resnet_stage(x, width=512, num_blocks=3, temporal_kernel=3, stride=(1, 2, 2), name="res5")
    # -> (T, H/32, W/32, 2048)

    x = keras.layers.GlobalAveragePooling3D(name="gap")(x)                    # -> (2048,)
    outputs = keras.layers.Dense(ENCODER_OUTPUT_DIM, name="feature_head")(x)  # -> (512,)

    return keras.Model(inputs, outputs, name="pose_encoder")


class PoseFeatureProjection(keras.layers.Layer):
    """(batch, 512) -> (batch, 384). Aligns the pose encoder's output with the
    planned DINOv3-based hand/face stream feature dimension (future work)."""

    def __init__(self, output_dim: int = PROJECTION_DIM, **kwargs):
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(output_dim, name="pose_projection")

    def call(self, x):
        return self.dense(x)


class PoseStream(keras.Model):
    """End-to-end pose stream: normalized keypoints -> 384-d pose feature.

        pose_stream = PoseStream(num_landmarks=49)
        pose_features = pose_stream(keypoints)  # (batch, T, J, 3) -> (batch, 384)

    Expects keypoints that are already body-normalized and hand-anchored (as
    produced by preprocessing.py, including anchor_hands_to_wrist) -- this
    class does not repeat either step.
    """

    def __init__(self, num_landmarks: int, height: int = HEATMAP_HEIGHT, width: int = HEATMAP_WIDTH,
                 sigma: float = HEATMAP_SIGMA, num_frames: int | None = None,
                 projection_dim: int = PROJECTION_DIM, **kwargs):
        super().__init__(**kwargs)
        self.heatmap_generator = HeatmapGenerator(height, width, sigma, channels_last=True)
        self.encoder = build_pose_encoder(num_landmarks, height, width, num_frames)
        self.projection = PoseFeatureProjection(projection_dim)

    def call(self, keypoints):
        # (batch, T, J, 3) -> (batch, T, H, W, J), channels-last for Conv3D
        heatmaps = self.heatmap_generator(keypoints)
        features = self.encoder(heatmaps)   # (batch, T, H, W, J) -> (batch, 512)
        return self.projection(features)    # (batch, 512) -> (batch, 384)

def demo():
    """Self-check: python signai/word_classification/models/keypoints_stream.py --self-check"""
    rng = np.random.default_rng(0)

    # 1) reshape_keypoints: (32, 147) -> (32, 49, 3)
    flat = rng.normal(size=(32, 147)).astype(np.float32)
    reshaped = reshape_keypoints(flat)
    assert reshaped.shape == (32, 49, 3)

    # 2) HeatmapGenerator, single clip: (32, 49, 3) -> (32, 49, 96, 96)
    heatmaps = HeatmapGenerator()(reshaped)
    assert heatmaps.shape == (32, 49, 96, 96)
    hm_np = heatmaps.numpy()
    assert hm_np.min() >= 0.0 and hm_np.max() <= 1.0

    # 3) batch + channels_last: (8, 32, 49, 3) -> (8, 32, 96, 96, 49)
    batch_kp = reshape_keypoints(rng.normal(size=(8, 32, 147)).astype(np.float32))
    batch_heatmaps = HeatmapGenerator(channels_last=True)(batch_kp)
    assert batch_heatmaps.shape == (8, 32, 96, 96, 49)

    # 4) pose encoder: (batch, 32, 96, 96, 49) -> (batch, 512)
    encoder = build_pose_encoder(num_landmarks=49, num_frames=32)
    encoded = encoder(batch_heatmaps)
    assert encoded.shape == (8, 512)

    # 5) projection: (batch, 512) -> (batch, 384)
    projection = PoseFeatureProjection()
    projected = projection(encoded)
    assert projected.shape == (8, 384)

    # 6) end-to-end PoseStream: (batch, 32, 49, 3) -> (batch, 384)
    pose_stream = PoseStream(num_landmarks=49, num_frames=32)
    raw_batch = reshape_keypoints(rng.normal(size=(8, 32, 147)).astype(np.float32))
    out = pose_stream(raw_batch)
    assert out.shape == (8, 384)
    assert np.all(np.isfinite(out.numpy()))

    print("self-check passed (6/6 checks)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true", help="run the self-check and exit")
    parser.add_argument("--visualize", action="store_true",
                         help="play skeleton+heatmap animation(s) for random real clip(s) (dataset/processed/test_data.npz)")
    parser.add_argument("--num-clips", type=int, default=1,
                         help="number of random clips to pull (with --visualize); side by side normally, "
                              "or stitched into one long clip with --combine")
    parser.add_argument("--combine", action="store_true",
                         help="stitch the --num-clips clips end to end into one long continuous playback "
                              "instead of showing them side by side, e.g. --num-clips 60 --combine for a longer clip")
    parser.add_argument("--fps", type=int, default=6, help="playback speed in frames/sec (with --visualize)")
    parser.add_argument("--heatmap-size", type=int, default=HEATMAP_HEIGHT,
                         help="heatmap height/width in pixels, to experiment with resolution (with --visualize, "
                              "default: %(default)s)")
    parser.add_argument("--smooth-hands", action="store_true",
                         help="apply an extra Savitzky-Golay smoothing pass to hand landmarks only, to reduce "
                              "MediaPipe hand-tracking jitter (with --visualize)")
    args = parser.parse_args()

    if args.self_check:
        demo()

    if args.visualize:
        clips = [reshape_keypoints(load_random_real_clip()).numpy() for _ in range(args.num_clips)]
        if args.combine:
            clips = [concat_real_clips(clips)]
        else:
            clips = [c[real_frame_indices(c)] for c in clips]
        if args.smooth_hands:
            clips = [smooth_hand_keypoints(c) for c in clips]
        viz = KeypointsVisualizer(height=args.heatmap_size, width=args.heatmap_size)
        viz.animate_clips(clips, fps=args.fps)
