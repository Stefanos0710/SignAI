from typing import Tuple

import numpy as np


class Augmentation:
    """Augmentation scaffold for encoder keypoint sequences.

    Planned augmentations:
    - Temporal
      - Linear Time Stretching: linear_time_stretch()
        -> The full video becomes uniformly faster or slower (factor 0.8 to 1.2).
      - Dynamic Time Warping: dynamic_time_warping()
        -> The speed changes within the video (for example, fast start, slow ending).
      - Frame Freeze: frame_freeze()
        -> A random frame is briefly repeated to simulate stutter.
      - Temporal Dropout: temporal_dropout()
        -> Random individual frames are removed (not full joints, only time steps).

    - Spatial
      - Random Shift: random_shift()
        -> The full person is shifted slightly left, right, up, or down.
      - Random Scaling: random_scaling()
        -> The person becomes slightly larger or smaller (zoom effect).
      - Z-Axis Rotation: z_axis_rotation()
        -> The upper body is tilted sideways by a small angle (about +/-5 degrees).
      - Point Noise: point_noise()
        -> Minimal Gaussian jitter is added to points to simulate tracking inaccuracy.
    """

    def __init__(
        # General settings
        self,
        seed: int = 42,
        augment_factor: int = 3,

        # Temporal: linear time stretch
        linear_stretch_min: float = 0.8,
        linear_stretch_max: float = 1.2,
        linear_stretch_probability: float = 0.5,

        # Temporal: dynamic time warping
        dynamic_warp_min: float = 0.8,
        dynamic_warp_max: float = 1.2,
        dynamic_warp_probability: float = 0.4,

        # Temporal: frame freeze
        frame_freeze_min: float = 1.0,
        frame_freeze_max: float = 3.0,
        frame_freeze_probability: float = 0.25,

        # Temporal: dropout
        temporal_dropout_min: float = 0.01,
        temporal_dropout_max: float = 0.08,
        temporal_dropout_probability: float = 0.35,

        # Spatial: random shift
        random_shift_min: float = -0.03,
        random_shift_max: float = 0.03,
        random_shift_probability: float = 0.3,

        # Spatial: random scaling
        random_scaling_min: float = 0.95,
        random_scaling_max: float = 1.05,
        random_scaling_probability: float = 0.3,

        # Spatial: z-axis rotation
        z_rotation_min: float = -5.0,
        z_rotation_max: float = 5.0,
        z_rotation_probability: float = 0.3,

        # Spatial: point noise
        point_noise_min: float = 0.0005,
        point_noise_max: float = 0.003,
        point_noise_probability: float = 0.3,

    ) -> None:
        # General config
        self.rng = np.random.default_rng(seed)
        self.seed = int(seed)
        self.augment_factor = max(0, int(augment_factor))

        # Temporal augmentation params
        self.linear_stretch_min = float(linear_stretch_min)
        self.linear_stretch_max = float(linear_stretch_max)
        self.linear_stretch_probability = float(linear_stretch_probability)

        self.dynamic_warp_min = float(dynamic_warp_min)
        self.dynamic_warp_max = float(dynamic_warp_max)
        self.dynamic_warp_probability = float(dynamic_warp_probability)

        self.frame_freeze_min = float(frame_freeze_min)
        self.frame_freeze_max = float(frame_freeze_max)
        self.frame_freeze_probability = float(frame_freeze_probability)

        self.temporal_dropout_min = float(temporal_dropout_min)
        self.temporal_dropout_max = float(temporal_dropout_max)
        self.temporal_dropout_probability = float(temporal_dropout_probability)

        # Spatial augmentation params
        self.random_shift_min = float(random_shift_min)
        self.random_shift_max = float(random_shift_max)
        self.random_shift_probability = float(random_shift_probability)

        self.random_scaling_min = float(random_scaling_min)
        self.random_scaling_max = float(random_scaling_max)
        self.random_scaling_probability = float(random_scaling_probability)

        self.z_rotation_min = float(z_rotation_min)
        self.z_rotation_max = float(z_rotation_max)
        self.z_rotation_probability = float(z_rotation_probability)

        self.point_noise_min = float(point_noise_min)
        self.point_noise_max = float(point_noise_max)
        self.point_noise_probability = float(point_noise_probability)

    # --- Temporal augmentations ---
    def linear_time_stretch(self, sequence: np.ndarray, factor: float = None) -> np.ndarray:
        """Placeholder for uniform temporal stretching."""
        pass

    def dynamic_time_warping(self, sequence: np.ndarray) -> np.ndarray:
        """Placeholder for non-uniform temporal warping."""
        pass

    def frame_freeze(self, sequence: np.ndarray) -> np.ndarray:
        """Placeholder for short frame repeat/stutter augmentation."""
        pass

    def temporal_dropout(self, sequence: np.ndarray) -> np.ndarray:
        """Placeholder for dropping random time steps."""
        pass

    # --- Spatial augmentations (placeholders) ---
    def random_shift(self, sequence: np.ndarray) -> np.ndarray:
        """Placeholder for global xy shift augmentation."""
        pass

    def random_scaling(self, sequence: np.ndarray) -> np.ndarray:
        """Placeholder for global scale augmentation."""
        pass

    def z_axis_rotation(self, sequence: np.ndarray) -> np.ndarray:
        """Placeholder for small z-axis body tilt augmentation."""
        pass

    def point_noise(self, sequence: np.ndarray) -> np.ndarray:
        """Placeholder for Gaussian keypoint noise."""
        pass

    def augment_training_split(
        self,
        encoder_data: np.ndarray,
        decoder_data: np.ndarray,
        target_data: np.ndarray,
        augment_factor: int = 1,
        keep_original: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return old + newly augmented training samples.

        Augmentation is applied only to encoder sequences; decoder/target labels are copied.
        """
        x_enc = np.asarray(encoder_data, dtype=np.float32)
        x_dec = np.asarray(decoder_data)
        y = np.asarray(target_data)

        if x_enc.ndim != 3:
            raise ValueError("encoder_data must be a 3D array: (samples, frames, features)")
        if x_enc.shape[0] != x_dec.shape[0] or x_enc.shape[0] != y.shape[0]:
            raise ValueError("encoder/decoder/target sample counts do not match")

        factor = max(0, int(augment_factor))
        if factor == 0:
            if keep_original:
                return x_enc, x_dec, y
            return (
                np.empty((0, x_enc.shape[1], x_enc.shape[2]), dtype=np.float32),
                np.empty((0, x_dec.shape[1]), dtype=x_dec.dtype),
                np.empty((0, y.shape[1]), dtype=y.dtype),
            )

        aug_enc = []
        aug_dec = []
        aug_y = []

        for i in range(x_enc.shape[0]):
            base_seq = x_enc[i]
            for _ in range(factor):
                # Placeholder behavior until augmentation methods are implemented.
                aug_enc.append(base_seq.copy())
                aug_dec.append(x_dec[i])
                aug_y.append(y[i])

        aug_enc_arr = np.asarray(aug_enc, dtype=np.float32)
        aug_dec_arr = np.asarray(aug_dec, dtype=x_dec.dtype)
        aug_y_arr = np.asarray(aug_y, dtype=y.dtype)

        if keep_original:
            return (
                np.concatenate([x_enc, aug_enc_arr], axis=0),
                np.concatenate([x_dec, aug_dec_arr], axis=0),
                np.concatenate([y, aug_y_arr], axis=0),
            )
        return aug_enc_arr, aug_dec_arr, aug_y_arr
