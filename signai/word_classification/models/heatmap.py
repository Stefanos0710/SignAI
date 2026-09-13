"""Per-landmark Gaussian heatmap rendering: keypoints (..., J, 3) -> heatmaps (..., J, H, W).

    heatmaps = HeatmapGenerator(height=96, width=96)(keypoints)

One self-contained unit: construct it once with the settings you want, call it
with keypoints, get heatmaps back. Split out of keypoints_stream.py so heatmap
rendering isn't tangled up with the pose encoder/model code.
"""
import numpy as np
import tensorflow as tf

HEATMAP_HEIGHT = 96
HEATMAP_WIDTH = 96
HEATMAP_SIGMA = 1.0
# Body-relative x/y (shoulder-distance units, from preprocessing.py's normalization)
# spans roughly +/-2.2 at the 99th percentile on the real dataset (checked
# empirically), not +/-0.5 -- COORD_EXTENT is the coordinate magnitude mapped to
# the canvas edge, so most of a clip's keypoints (including raised hands) actually
# land on the heatmap instead of off-canvas.
COORD_EXTENT = 3.0


class HeatmapGenerator:
    """Renders one Gaussian heatmap per landmark from normalized (x, y) keypoints.

    z is never used (no position, no amplitude) -- only x/y determine the heatmap.
    Output is always in [0, 1] since it's just exp(<=0). channels_last=True gives
    (..., H, W, J) instead of (..., J, H, W), for feeding a Conv3D.
    """

    def __init__(self, height: int = HEATMAP_HEIGHT, width: int = HEATMAP_WIDTH,
                 sigma: float = HEATMAP_SIGMA, coord_extent: float = COORD_EXTENT,
                 channels_last: bool = False):
        self.height = height
        self.width = width
        self.sigma = sigma
        self.coord_extent = coord_extent
        self.channels_last = channels_last

    def __call__(self, keypoints):
        keypoints = tf.convert_to_tensor(keypoints, dtype=tf.float32)
        x, y = keypoints[..., 0], keypoints[..., 1]  # each (..., J), z is unused

        px = (x / (2.0 * self.coord_extent) + 0.5) * self.width
        py = (y / (2.0 * self.coord_extent) + 0.5) * self.height

        heatmaps = self._render(px, py)  # (..., J, H, W)

        if self.channels_last:
            # (..., J, H, W) -> (..., H, W, J), e.g. (B, T, J, H, W) -> (B, T, H, W, J)
            rank = heatmaps.shape.rank
            perm = list(range(rank - 3)) + [rank - 2, rank - 1, rank - 3]
            heatmaps = tf.transpose(heatmaps, perm)

        return heatmaps

    def _render(self, px, py):
        """px, py: (...,) pixel coordinates. Returns (..., height, width), vectorized --
        broadcasts a 1D grid against the leading (...) dims, no loop over points, frames,
        or batch. Same technique as preprocessing.py::generate_heatmap.
        """
        grid_y = tf.range(self.height, dtype=tf.float32)  # (H,)
        grid_x = tf.range(self.width, dtype=tf.float32)   # (W,)

        dy2 = (grid_y - py[..., None]) ** 2  # (..., H)
        dx2 = (grid_x - px[..., None]) ** 2  # (..., W)

        # (..., H, 1) + (..., 1, W) -> (..., H, W)
        sq_dist = dy2[..., :, None] + dx2[..., None, :]
        return tf.exp(-sq_dist / (2.0 * self.sigma ** 2))


def demo():
    """Self-check: python signai/word_classification/models/heatmap.py --self-check"""
    rng = np.random.default_rng(0)

    # single clip: (32, 49, 3) -> (32, 49, 96, 96), values in [0, 1]
    keypoints = rng.normal(size=(32, 49, 3)).astype(np.float32)
    heatmaps = HeatmapGenerator()(keypoints).numpy()
    assert heatmaps.shape == (32, 49, 96, 96)
    assert heatmaps.min() >= 0.0 and heatmaps.max() <= 1.0

    # batch + channels_last: (8, 32, 49, 3) -> (8, 32, 96, 96, 49)
    batch = rng.normal(size=(8, 32, 49, 3)).astype(np.float32)
    batch_heatmaps = HeatmapGenerator(channels_last=True)(batch)
    assert batch_heatmaps.shape == (8, 32, 96, 96, 49)

    # a keypoint at (0, 0) (shoulder midpoint) lands at canvas center
    center_kp = np.zeros((1, 3), dtype=np.float32)
    hm = HeatmapGenerator(height=32, width=32)(center_kp).numpy()[0]
    assert np.unravel_index(np.argmax(hm), hm.shape) == (16, 16)

    print("self-check passed (3/3 checks)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true", help="run the self-check and exit")
    args = parser.parse_args()
    if args.self_check:
        demo()
