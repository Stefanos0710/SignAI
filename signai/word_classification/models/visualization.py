"""Debug/visualization helpers for the pose keypoints stream: static heatmap+
skeleton snapshots and animated playback, plus the dataset-loading/frame-index
utilities they share. Not used by training or inference -- CLI only
(keypoints_stream.py --visualize).
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from signai.word_classification.models.heatmap import (  # noqa: E402
    HeatmapGenerator, HEATMAP_HEIGHT, HEATMAP_WIDTH, HEATMAP_SIGMA, COORD_EXTENT,
)
from signai.word_classification.paths import PROCESSED_DIR  # noqa: E402

DATASET_DIR = str(PROCESSED_DIR)

_POSE_LANDMARK_NAMES = [
    "nose", "L_shoulder", "R_shoulder", "L_elbow", "R_elbow", "L_wrist", "R_wrist",
]
# (from, to) pairs of _POSE_LANDMARK_NAMES indices, drawn as skeleton lines so the
# body layout reads at a glance instead of as an unstructured point cloud.
_POSE_BONES = [(0, 1), (0, 2), (1, 2), (1, 3), (3, 5), (2, 4), (4, 6)]


class KeypointsVisualizer:
    """Renders normalized (T, J, 3) pose keypoint clips as heatmap+skeleton
    views, static or animated. Mirrors preprocessing.py::test_display_heatmap_frame().

    viz = KeypointsVisualizer()
    viz.visualize_heatmaps(keypoints)          # single frame, static
    viz.animate_clips(clips)                    # full clip(s), animated
    """

    def __init__(self, height: int = HEATMAP_HEIGHT, width: int = HEATMAP_WIDTH,
                 sigma: float = HEATMAP_SIGMA, coord_extent: float = COORD_EXTENT):
        self.height = height
        self.width = width
        self.sigma = sigma
        self.coord_extent = coord_extent

    def _pixel_coords(self, kp):
        """(J, 3) real-space keypoints -> (px, py) pixel coordinates for plotting."""
        px = (kp[:, 0] / (2.0 * self.coord_extent) + 0.5) * self.width
        py = (kp[:, 1] / (2.0 * self.coord_extent) + 0.5) * self.height
        return px, py

    def _style_axes(self, ax):
        ax.set_xlim(0, self.width)
        ax.set_ylim(self.height, 0)  # inverted, matches imshow's origin="upper"
        ax.set_aspect("equal")

    def visualize_heatmaps(self, keypoints, frame_idx: int = 0, save_path=None):
        """Debug helper: render one clip's heatmaps, max-project across landmarks
        for a single frame, and overlay the actual keypoint positions as a
        scatter plot, to visually confirm keypoints land where they should.

        Pose/left-hand/right-hand landmarks are colored and the 7 pose
        landmarks are labeled -- an unlabeled max-projection of an asymmetric
        two-hand pose (very common in sign language, e.g. one hand raised near
        the face, the other at the hip) is easy to misread as
        "rotated"/"sideways" when every point looks the same.

        keypoints: (T, J, 3) for one clip, already body-normalized. frame_idx
        should point at a real (non-padding) frame -- clips shorter than T are
        zero-padded, and an all-zero frame collapses every landmark onto a
        single center pixel.
        """
        import matplotlib.pyplot as plt

        keypoints_np = np.asarray(keypoints, dtype=np.float32)
        if not np.any(keypoints_np[frame_idx]):
            raise ValueError(f"frame_idx={frame_idx} is an all-zero (padding) frame, pick a real frame")

        heatmaps = HeatmapGenerator(self.height, self.width, self.sigma, self.coord_extent)(keypoints_np).numpy()  # (T, J, H, W)

        frame_heatmaps = heatmaps[frame_idx]   # (J, H, W)
        projected = frame_heatmaps.max(axis=0)  # (H, W), max over landmarks

        frame_kp = keypoints_np[frame_idx]      # (J, 3)
        px, py = self._pixel_coords(frame_kp)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        ax1.imshow(projected, cmap="hot", vmin=0.0, vmax=1.0)
        ax1.set_title(f"frame {frame_idx}: heatmaps (max over {frame_heatmaps.shape[0]} landmarks)")
        ax2.set_facecolor("black")
        ax2.set_title("skeleton (clean view)")

        for ax in (ax1, ax2):
            ax.scatter(px[7:28], py[7:28], s=14, c="orange", label="left hand")
            ax.scatter(px[28:49], py[28:49], s=14, c="deepskyblue", label="right hand")
            for a, b in _POSE_BONES:
                ax.plot([px[a], px[b]], [py[a], py[b]], c="white", linewidth=2, zorder=2)
            ax.scatter(px[0:7], py[0:7], s=60, c="white", edgecolors="black", linewidths=1, zorder=3, label="pose")
            self._style_axes(ax)

        for i, name in enumerate(_POSE_LANDMARK_NAMES):
            ax2.annotate(name, (px[i], py[i]), xytext=(3, 3), textcoords="offset points",
                         color="black", fontsize=8, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
        ax2.legend(fontsize=8, loc="upper right")

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def animate_clips(self, clips, fps: int = 6, save_path=None):
        """Debug helper: play a clip's skeleton + heatmap over time, i.e. the
        same view as visualize_heatmaps but animated across all of the clip's
        real (non-padding) frames instead of a single static one.

        clips: a single (T, J, 3) clip, or a list of them -- each gets its own
        subplot, animating over just its own real frames (independent lengths,
        shorter clips loop). Useful for eyeballing several signs side by side.
        Interactive (plt.show()) unless save_path is given (e.g. "clips.gif",
        needs pillow installed; "clips.mp4" needs ffmpeg).
        """
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        if isinstance(clips, np.ndarray):
            clips = [clips]
        clips = [np.asarray(c, dtype=np.float32) for c in clips]

        real_frames = [real_frame_indices(c) for c in clips]
        if any(len(f) == 0 for f in real_frames):
            raise ValueError("one of the clips has no non-padding frames")
        num_steps = max(len(f) for f in real_frames)

        fig, axes = plt.subplots(1, len(clips), figsize=(5 * len(clips), 5), squeeze=False)
        axes = axes[0]

        panels = []
        for i, (ax, clip, frames) in enumerate(zip(axes, clips, real_frames)):
            heatmaps = HeatmapGenerator(self.height, self.width, self.sigma, self.coord_extent)(clip).numpy()  # (T, J, H, W)
            img = ax.imshow(heatmaps[frames[0]].max(axis=0), cmap="hot", vmin=0.0, vmax=1.0)
            bone_lines = [ax.plot([], [], c="white", linewidth=2, zorder=2)[0] for _ in _POSE_BONES]
            left_hand = ax.scatter([], [], s=14, c="orange", label="left hand")
            right_hand = ax.scatter([], [], s=14, c="deepskyblue", label="right hand")
            pose_pts = ax.scatter([], [], s=60, c="white", edgecolors="black", linewidths=1, zorder=3, label="pose")
            self._style_axes(ax)
            panels.append(dict(idx=i, ax=ax, clip=clip, frames=frames, heatmaps=heatmaps, img=img,
                                bone_lines=bone_lines, left_hand=left_hand, right_hand=right_hand, pose_pts=pose_pts))
        axes[0].legend(fontsize=8, loc="upper right")

        def update(step):
            updated = []
            for p in panels:
                t = p["frames"][step % len(p["frames"])]
                px, py = self._pixel_coords(p["clip"][t])

                p["img"].set_data(p["heatmaps"][t].max(axis=0))
                for line, (a, b) in zip(p["bone_lines"], _POSE_BONES):
                    line.set_data([px[a], px[b]], [py[a], py[b]])
                p["left_hand"].set_offsets(np.stack([px[7:28], py[7:28]], axis=1))
                p["right_hand"].set_offsets(np.stack([px[28:49], py[28:49]], axis=1))
                p["pose_pts"].set_offsets(np.stack([px[0:7], py[0:7]], axis=1))
                p["ax"].set_title(f"clip {p['idx']}: frame {t}")
                updated += [p["img"], *p["bone_lines"], p["left_hand"], p["right_hand"], p["pose_pts"]]
            return updated

        ani = animation.FuncAnimation(fig, update, frames=num_steps, interval=1000 / fps, blit=False)

        if save_path:
            ani.save(save_path, fps=fps)
            plt.close(fig)
        else:
            plt.show()

        return ani


def real_frame_indices(keypoints):
    """Indices of non-padding frames in a (T, J, 3) clip (clips shorter than T
    are zero-padded, so not every frame index has real keypoints in it)."""
    return [t for t in range(keypoints.shape[0]) if np.any(keypoints[t])]


def random_nonpadded_frame_idx(keypoints):
    """Index of a random non-padding frame in a (T, J, 3) clip."""
    real = real_frame_indices(keypoints)
    if not real:
        raise ValueError("clip has no non-padding frames")
    return int(np.random.choice(real))


def concat_real_clips(clips):
    """Stitch several (T, J, 3) clips' real (non-padding) frames end to end into
    one longer (T_total, J, 3) sequence -- e.g. many short sign clips concatenated
    into one continuous "video" to play back with animate_clips, since a single
    clip on its own is only ~1 real second long."""
    return np.concatenate(
        [np.asarray(c, dtype=np.float32)[real_frame_indices(c)] for c in clips], axis=0
    )


def load_random_real_clip(split: str = "test"):
    """Pick one random clip's keypoints, (32, 147), from the real on-disk dataset
    (dataset/processed/{split}_data.npz) -- for the visualizer's CLI demo.
    Already body-normalized by preprocessing.py, no further normalization needed.
    """
    path = os.path.join(DATASET_DIR, f"{split}_data.npz")
    X = np.load(path, allow_pickle=True)["X"]  # (N, 32, 147)
    return X[np.random.randint(len(X))]
