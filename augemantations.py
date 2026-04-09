from typing import Optional, Tuple
from scipy.interpolate import interp1d
import numpy as np


class Augmentation:
    """Augmentation scaffold for encoder keypoint sequences.

    Augmentations:
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
        augment_factor: int = 3, # Number of augmented samples to create per original sample (0 = no augmentation, 1 = 1 new aug sample set, etc.)
        keep_original: bool = True,

        # Temporal: linear time stretch
        linear_stretch_min: float = 0.8,
        linear_stretch_max: float = 1.2,
        linear_stretch_probability: float = 0.5,

        # Temporal: dynamic time warping
        dynamic_warp_min: float = 0.9,
        dynamic_warp_max: float = 1.1,
        dynamic_warp_probability: float = 0.3,

        # Temporal: frame freeze
        frame_freeze_min: float = 1.0,
        frame_freeze_max: float = 2.0,
        frame_freeze_frequency_max: float = 1, # max number of freezes per sequence
        frame_freeze_probability: float = 0.15,

        # Temporal: dropout
        temporal_dropout_min: float = 0.05,
        temporal_dropout_max: float = 0.10,
        temporal_dropout_probability: float = 0.2,

        # Spatial: random shift
        random_shift_min: float = -0.03,
        random_shift_max: float = 0.03,
        random_shift_probability: float = 0.4,

        # Spatial: random scaling
        random_scaling_min: float = 0.92,
        random_scaling_max: float = 1.08,
        random_scaling_probability: float = 0.4,

        # Spatial: z-axis rotation
        z_rotation_min: float = -4.0,
        z_rotation_max: float = 4.0,
        z_rotation_probability: float = 0.3,

        # Spatial: point noise
        point_noise_min: float = 0.0005,
        point_noise_max: float = 0.002,
        point_noise_probability: float = 0.5,

    ) -> None:
        # General config
        self.rng = np.random.default_rng(seed)
        self.seed = int(seed)
        self.augment_factor = max(0, int(augment_factor))
        self.keep_original = bool(keep_original)

        # Temporal augmentation params
        self.linear_stretch_min = float(linear_stretch_min)
        self.linear_stretch_max = float(linear_stretch_max)
        self.linear_stretch_probability = float(linear_stretch_probability)

        self.dynamic_warp_min = float(dynamic_warp_min)
        self.dynamic_warp_max = float(dynamic_warp_max)
        self.dynamic_warp_probability = float(dynamic_warp_probability)

        self.frame_freeze_min = float(frame_freeze_min)
        self.frame_freeze_max = float(frame_freeze_max)
        self.frame_freeze_frequency_max = float(frame_freeze_frequency_max)
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

    def _normalize_sequence_shape(
        self,
        sequence: np.ndarray,
        target_timesteps: int,
        target_features: int,
    ) -> np.ndarray:
        """Normalize a sequence to fixed [T, F] with strict right-padding (zeros at the end only)."""
        seq = np.asarray(sequence, dtype=np.float32)
        if seq.ndim != 2:
            raise ValueError(f"Expected 2D sequence [T, F], got shape={seq.shape}")

        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)

        if seq.shape[1] != target_features:
            if seq.shape[1] > target_features:
                seq = seq[:, :target_features]
            else:
                pad_w = target_features - seq.shape[1]
                seq = np.pad(seq, ((0, 0), (0, pad_w)), mode='constant', constant_values=0.0)

        if seq.shape[0] == 0:
            seq = np.zeros((1, target_features), dtype=np.float32)

        if seq.shape[0] > target_timesteps:
            seq = seq[:target_timesteps]
        elif seq.shape[0] < target_timesteps:
            pad_t = target_timesteps - seq.shape[0]
            seq = np.pad(seq, ((0, pad_t), (0, 0)), mode='constant', constant_values=0.0)

        return seq.astype(np.float32, copy=False)

    def _strip_right_padding(self, sequence: np.ndarray) -> np.ndarray:
        """Remove trailing all-zero frames so temporal augmentation only touches valid frames."""
        seq = np.asarray(sequence, dtype=np.float32)
        if seq.ndim != 2:
            raise ValueError(f"Expected 2D sequence [T, F], got shape={seq.shape}")

        non_zero_rows = np.any(np.abs(seq) > 1e-6, axis=1)
        if not np.any(non_zero_rows):
            return seq[:1].copy()

        last_valid = int(np.where(non_zero_rows)[0][-1]) + 1
        return seq[:last_valid].copy()

    # --- Temporal augmentations ---
    def linear_time_stretch(self, sequence: np.ndarray) -> np.ndarray:
        """
        In this function, we apply a linear time stretch to the input sequence by resampling it to a new length determined by a random stretch factor set in the config self settings.

        This is the step-by-step process:
            1. Get the stretch factor by getting a random number between self.linear_stretch_min and self.linear_stretch_max.

            2. Create the old time steps (old_x) as a linear grid from 0 to 1 across the original number of frames.

            3. Create the new time steps (new_x) as a linear grid from 0 to 1 across the new number of frames determined by multiplying the original number of frames by the stretch factor.

            4. By using the scipy interp1d function (linear and in the time dimension only), we interpolate the original sequence at the new time steps to get the stretched sequence. The interpolation function is created for each feature dimension separately.

            5. At last, we return the stretched sequence in float32 format to ensure it matches the data type of the original input sequence.
        """

        # get random stretch factor for this sequence
        stretch_factor = self.rng.uniform(self.linear_stretch_min, self.linear_stretch_max)

        # get original number of frames
        num_frames = sequence.shape[0]

        # create the old time steps
        # old grid goes linearly from 0 to 1 across the original frames
        old_x = np.linspace(0, 1, num_frames)

        # create the new time steps by applying the stretch factor
        # new grid goes linearly from 0 to 1 across the new stretched frames
        new_x = np.linspace(0, 1 / stretch_factor, num_frames)

        # make sure new_x is within the range of old_x for interpolation
        new_X = np.clip(new_x, 0, 1)

        # interpolate each feature dimension separately
        interpolation_func = interp1d(old_x, sequence, axis=0, kind='linear', fill_value='extrapolate')
        stretched_sequence = interpolation_func(new_X)
        
        # return the stretched sequence in float32 format to ensure matching data type
        return stretched_sequence.astype(np.float32)

    def dynamic_time_warping(self, sequence: np.ndarray) -> np.ndarray:
        """
        In this function, we apply a dynamic time warping to the input sequence by creating a smooth nonlinear time map, so that different parts of the sequence can be stretched or compressed with different local speeds.

        This is the step-by-step process:
            1. Get the number of frames and create the reference time grid (target_indices) as a linear grid from 0 to 1 across the original sequence length.

            2. Create four control points (start, two middle points, end) that define the base time map.

            3. Create random noise for the two middle control points by using self.dynamic_warp_min and self.dynamic_warp_max, then add this noise to the control points.

            4. Clamp and sort the warped control points to keep them valid in the range [0, 1], and force the first and last point to exactly 0 and 1.

            5. Build a smooth quadratic interpolation curve (warp_fn) from the original control points to the warped control points, and use it to generate warped indices for every frame.

            6. Clip the warped indices to [0, 1], then use scipy interp1d (linear in the time dimension) to resample the original sequence at these warped indices.

            7. At last, we return the warped sequence in float32 format to ensure it matches the data type of the original input sequence.
        """

        num_frames = sequence.shape[0]
        
        # 1. Create the reference grid (0 to 1)
        # This represents our target frame indices
        target_indices = np.linspace(0, 1, num_frames)

        # 2. Create 4 control points (Start, 2x Middle, End)
        control_points = np.linspace(0, 1, 4)

        # 3. Create and apply noise only to the 2 middle control points
        # Use self.dynamic_warp_max to control the intensity
        noise = np.zeros(4)
        noise[1:3] = self.rng.uniform(self.dynamic_warp_min, self.dynamic_warp_max, size=2)

        # Apply noise and ensure the time stays valid (0 to 1 and strictly increasing)
        warped_control_points = control_points + noise
        warped_control_points = np.clip(warped_control_points, 0, 1)
        warped_control_points = np.sort(warped_control_points)
        warped_control_points[0] = 0.0  # Force start at 0
        warped_control_points[-1] = 1.0 # Force end at 1

        # 4. Create the Warping Curve (The "Bent Ruler")
        # We use 'quadratic' to ensure smooth acceleration/deceleration
        warp_fn = interp1d(control_points, warped_control_points, kind='quadratic')
        
        # This calculates where each target frame should "look" in the original sequence
        warped_indices = warp_fn(target_indices)
        warped_indices = np.clip(warped_indices, 0, 1) # Safety clip

        # 5. Interpolate the actual sequence data
        # We use 'linear' for the keypoints to avoid overshoot artifacts
        interpolation_func = interp1d(target_indices, sequence, axis=0, kind='linear', fill_value='extrapolate')
        
        # Apply the warped time-map to the original data
        warped_sequence = interpolation_func(warped_indices)

        # Return in float32 format to match the input type
        return warped_sequence.astype(np.float32)

    def frame_freeze(self, sequence: np.ndarray) -> np.ndarray:  
        """
        In this function, we apply a frame freeze augmentation to the input sequence by repeating one or more randomly selected frames for a short random duration, which simulates brief stutters in temporal motion.

        This is the step-by-step process:
            1. First, we draw a random number and compare it to self.frame_freeze_probability to decide whether the augmentation is applied.

            2. If the augmentation is enabled, we create a copy of the input sequence so the original sequence is not modified in-place.

            3. We determine how many freeze events should be applied (num_freezes), and for each event we recompute the current sequence length because the sequence becomes longer after each insertion.

            4. For each freeze event, we sample a random freeze duration between self.frame_freeze_min and self.frame_freeze_max, and select a random frame index to freeze.

            5. We extract the selected frame, repeat it for the sampled duration, and insert the repeated frames back into the sequence at the chosen position.

            6. This process increases the total number of frames and introduces local temporal pauses that mimic short capture lags or motion stalls.

            7. At last, we return the augmented sequence in float32 format to ensure it matches the data type of the original input sequence.
        """

        # work on a copy to avoid modifying the original data
        sequence = sequence.copy()
        
        # get the number of freezes to apply (frequency)
        num_freezes = int(self.frame_freeze_frequency_max)

        for _ in range(num_freezes):
            # get current number of frames (important because sequence grows each loop)
            num_frames = sequence.shape[0]

            # get freeze duration in frames (random between min and max)
            freeze_duration = self.rng.integers(self.frame_freeze_min, self.frame_freeze_max + 1)

            # find random frame index to freeze
            freeze_index = self.rng.integers(0, num_frames)

            # create the frozen sequence by inserting the frozen frame
            # we take one frame as a slice to keep the (1, features) shape
            frozen_frame = sequence[freeze_index : freeze_index + 1]
            
            # split the sequence and insert the repeated frame
            before = sequence[:freeze_index]
            after = sequence[freeze_index:]
            
            # concatenate everything back together (sequence length increases here)
            sequence = np.concatenate([
                before, 
                np.repeat(frozen_frame, freeze_duration, axis=0), 
                after
            ], axis=0)

        # return the longer sequence in float32 format
        return sequence.astype(np.float32)

    def temporal_dropout(self, sequence: np.ndarray) -> np.ndarray:
        """
        In this function, we apply temporal dropout to the input sequence by removing a random set of frames, which simulates small temporal discontinuities like skipped capture moments.

        This is the step-by-step process:
            1. First, we get the total number of frames in the sequence.

            2. We sample how many frames should be dropped (num_to_drop) between self.temporal_dropout_min and self.temporal_dropout_max.

            3. We run a safety check so we do not remove too many frames, because we still need a minimum sequence length for valid processing.

            4. We sample unique frame indices to drop, so the same frame index is not selected multiple times.

            5. We build a boolean keep mask, mark selected indices as False, and keep all other frames.

            6. At last, we return the shortened sequence in float32 format to keep dtype consistency for the pipeline.
        """
        num_frames = sequence.shape[0]
        
        # determine how many frames to drop.
        # If values are <= 1.0, interpret as percentage range; otherwise as absolute frame counts.
        if self.temporal_dropout_max <= 1.0:
            drop_ratio = float(self.rng.uniform(self.temporal_dropout_min, self.temporal_dropout_max))
            num_to_drop = int(round(num_frames * drop_ratio))
            num_to_drop = max(1, num_to_drop)
        else:
            min_drop = int(round(self.temporal_dropout_min))
            max_drop = int(round(self.temporal_dropout_max))
            if max_drop < min_drop:
                min_drop, max_drop = max_drop, min_drop
            num_to_drop = int(self.rng.integers(min_drop, max_drop + 1))
        
        # safety check: we still need at least a few frames to keep the sequence valid
        if num_to_drop >= num_frames - 2:
            return sequence

        # get random indices to drop (unique so we do not drop the same index twice)
        drop_indices = self.rng.choice(np.arange(num_frames), size=num_to_drop, replace=False)

        # create a keep mask for all frames except the selected drop indices
        keep_mask = np.ones(num_frames, dtype=bool)
        keep_mask[drop_indices] = False

        # apply the mask to the sequence (sequence length decreases here)
        dropped_sequence = sequence[keep_mask]

        # return the shorter sequence in float32 format
        return dropped_sequence.astype(np.float32)


    # --- Spatial augmentations ---
    def random_shift(self, sequence: np.ndarray) -> np.ndarray:
        """
        In this function, we apply a random spatial shift to the full person by moving all keypoints in x and y direction, which simulates camera framing changes like slight left/right/up/down offsets.

        This is the step-by-step process:
            1. First, we create a copy of the input sequence so the original data is not modified in-place.

            2. We sample two random shift values (shift_x and shift_y) between self.random_shift_min and self.random_shift_max.

            3. Since the input features are arranged as (x, y, z), we add shift_x to all x coordinates and shift_y to all y coordinates.

            4. At last, we return the shifted sequence in float32 format to ensure it matches the expected input/output type.
        """
        sequence = sequence.copy()
        
        # sample random values for the x and y shift
        shift_x = self.rng.uniform(self.random_shift_min, self.random_shift_max)
        shift_y = self.rng.uniform(self.random_shift_min, self.random_shift_max)

        # since your data layout is (x, y, z):
        sequence[:, 0::3] += shift_x  # all x coordinates
        sequence[:, 1::3] += shift_y  # all y coordinates

        return sequence.astype(np.float32)

    def random_scaling(self, sequence: np.ndarray) -> np.ndarray:
        """
        In this function, we apply a random spatial scaling to simulate a zoom effect, so the full person appears slightly larger or smaller while staying centered in the frame.

        This is the step-by-step process:
            1. First, we create a copy of the input sequence to avoid changing the original sequence in-place.

            2. We sample a random scale factor between self.random_scaling_min and self.random_scaling_max.

            3. We define the frame center as 0.5 and scale all x and y coordinates around this center point.

            4. At last, we return the scaled sequence in float32 format so it stays consistent with the expected model input type.
        """

        sequence = sequence.copy()
        scale = self.rng.uniform(self.random_scaling_min, self.random_scaling_max)

        # we scale around the center point of the frame (0.5)
        center = 0.5
        
        sequence[:, 0::3] = (sequence[:, 0::3] - center) * scale + center
        sequence[:, 1::3] = (sequence[:, 1::3] - center) * scale + center

        return sequence.astype(np.float32)

    def z_axis_rotation(self, sequence: np.ndarray) -> np.ndarray:
        """
        In this function, we apply a small rotation around the z-axis by rotating all (x, y) keypoint pairs around the frame center, which simulates slight sideways upper-body tilt.

        This is the step-by-step process:
            1. First, we copy the sequence so the original sequence is kept unchanged.

            2. We sample a random rotation angle in degrees between self.z_rotation_min and self.z_rotation_max, then convert it to radians.

            3. We compute cosine and sine of the angle once, and define 0.5 as the center of rotation.

            4. We iterate through the feature vector in (x, y, z) layout, take each (x, y) pair, shift it to center-based coordinates, apply the 2D rotation matrix, and shift it back.

            5. At last, we return the rotated sequence in float32 format to keep a stable numeric dtype for the pipeline.
        """

        sequence = sequence.copy()
        
        # convert random angle from degree to radian
        angle_rad = np.radians(self.rng.uniform(self.z_rotation_min, self.z_rotation_max))
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        center = 0.5
        
        # apply rotation to each (x, y) pair
        for i in range(0, sequence.shape[1], 3):
            x = sequence[:, i] - center
            y = sequence[:, i+1] - center
            
            sequence[:, i] = (x * cos_a - y * sin_a) + center
            sequence[:, i+1] = (x * sin_a + y * cos_a) + center

        return sequence.astype(np.float32)

    def point_noise(self, sequence: np.ndarray) -> np.ndarray:
        """
        In this function, we add small Gaussian point noise to the full sequence so each keypoint gets a minimal jitter, which simulates measurement/tracking inaccuracy.

        This is the step-by-step process:
            1. We sample a random noise standard deviation (noise_sigma) between self.point_noise_min and self.point_noise_max.

            2. We generate Gaussian noise with mean 0 and the sampled sigma for the full sequence shape (frames, features).

            3. We add this noise to the original sequence values.

            4. At last, we return the noisy sequence in float32 format to keep dtype compatibility with the rest of the training pipeline.
        """

        # generate noise for the full array (frames, features)
        noise_sigma = self.rng.uniform(self.point_noise_min, self.point_noise_max)
        noise = self.rng.normal(0, noise_sigma, size=sequence.shape)

        return (sequence + noise).astype(np.float32)

    # add the pipline and the probability logic to the main augmentation function
    def pipeline_augment(self, sequence: np.ndarray) -> np.ndarray:
        """Apply a random combination of augmentations to the input sequence based on the configured probabilities."""

        sequence = np.asarray(sequence, dtype=np.float32)
        if sequence.ndim != 2:
            raise ValueError(f"Expected 2D sequence [T, F], got shape={sequence.shape}")

        target_timesteps = int(sequence.shape[0])
        target_features = int(sequence.shape[1])

        # Start from guaranteed right-padded shape and only augment the valid prefix.
        sequence = self._normalize_sequence_shape(sequence, target_timesteps, target_features)
        sequence = self._strip_right_padding(sequence)

        # --- Temporal ---
        if self.rng.random() < self.linear_stretch_probability:
            sequence = self.linear_time_stretch(sequence)
            sequence = self._normalize_sequence_shape(sequence, target_timesteps, target_features)
            sequence = self._strip_right_padding(sequence)

        if self.rng.random() < self.dynamic_warp_probability:
            sequence = self.dynamic_time_warping(sequence)
            sequence = self._normalize_sequence_shape(sequence, target_timesteps, target_features)
            sequence = self._strip_right_padding(sequence)

        if self.rng.random() < self.frame_freeze_probability:
            sequence = self.frame_freeze(sequence)
            sequence = self._normalize_sequence_shape(sequence, target_timesteps, target_features)
            sequence = self._strip_right_padding(sequence)

        if self.rng.random() < self.temporal_dropout_probability:
            sequence = self.temporal_dropout(sequence)
            sequence = self._normalize_sequence_shape(sequence, target_timesteps, target_features)
            sequence = self._strip_right_padding(sequence)

        # --- Spatial ---
        if self.rng.random() < self.random_shift_probability:
            sequence = self.random_shift(sequence)

        if self.rng.random() < self.random_scaling_probability:
            sequence = self.random_scaling(sequence)

        if self.rng.random() < self.z_rotation_probability:
            sequence = self.z_axis_rotation(sequence)
            
        if self.rng.random() < self.point_noise_probability:
            sequence = self.point_noise(sequence)

        return self._normalize_sequence_shape(sequence, target_timesteps, target_features)

    def augment_training_split(
        self,
        encoder_data: np.ndarray,
        decoder_data: np.ndarray,
        target_data: np.ndarray,
        augment_factor: Optional[int] = None, # makes it possible to override the number of augmentations per sample for this specific call, otherwise it uses the default from __init__
        keep_original: Optional[bool] = None  # makes it possible to override the keep_original setting for this specific call, otherwise it uses the default from __init__
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return old + newly augmented training samples."""
        
        # 1. set up augmentation parameters for this call (use defaults if not provided)
        factor = self.augment_factor if augment_factor is None else max(0, int(augment_factor))
        keep = self.keep_original if keep_original is None else bool(keep_original)

        x_enc = np.asarray(encoder_data, dtype=np.float32)
        x_dec = np.asarray(decoder_data)
        y = np.asarray(target_data)

        aug_enc = []
        aug_dec = []
        aug_y = []
        target_timesteps = int(x_enc.shape[1])
        target_features = int(x_enc.shape[2])

        # 2. add the augmentation samples
        for i in range(x_enc.shape[0]):
            base_seq = x_enc[i]
            
            # if keep_original is True, we add the original sample to the augmented dataset before creating new versions
            if keep:
                aug_enc.append(self._normalize_sequence_shape(base_seq, target_timesteps, target_features))
                aug_dec.append(x_dec[i])
                aug_y.append(y[i])

            # generate 'factor' augmented versions of the current sample and add them to the augmented dataset
            for _ in range(factor):
                # HIER: Deine Pipeline aufrufen statt nur .copy()
                augmented = self.pipeline_augment(base_seq.copy())
                
                aug_enc.append(self._normalize_sequence_shape(augmented, target_timesteps, target_features))
                aug_dec.append(x_dec[i])
                aug_y.append(y[i])

        return np.asarray(aug_enc, dtype=np.float32), np.asarray(aug_dec), np.asarray(aug_y)
