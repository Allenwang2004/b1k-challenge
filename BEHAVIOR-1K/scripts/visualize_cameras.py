#!/usr/bin/env python3
"""
Extract sample frames from all three cameras in the recorded dataset
and save a side-by-side grid image.

Usage:
  python scripts/visualize_cameras.py [--dataset data/torque_aug_v0] [--out /tmp/cameras.png]
"""
import argparse
import os
import cv2
import numpy as np

CAMERAS = [
    ("Head (ZED)", "observation.rgb.zed_link_camera_0"),
    ("Left wrist", "observation.rgb.left_realsense_link_camera_0"),
    ("Right wrist", "observation.rgb.right_realsense_link_camera_0"),
]

N_SAMPLES = 6  # frames to show per camera


def extract_frames(mp4_path: str, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(mp4_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [int(i * (total - 1) / (n - 1)) for i in range(n)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


def make_grid(dataset_root: str, out_path: str, scale: int = 4) -> None:
    rows = []
    for label, key in CAMERAS:
        mp4 = os.path.join(dataset_root, "videos", key, "chunk-000", "file-000.mp4")
        if not os.path.exists(mp4):
            print(f"  [skip] {mp4} not found")
            continue
        frames = extract_frames(mp4, N_SAMPLES)
        # Upscale tiny 64×64 frames so they're visible
        frames = [cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                  for f in frames]
        # Label each frame with its index
        total_w = sum(f.shape[1] for f in frames) + 4 * (len(frames) - 1)
        h = frames[0].shape[0]
        row = np.zeros((h, total_w, 3), dtype=np.uint8)
        x = 0
        for i, f in enumerate(frames):
            w = f.shape[1]
            row[:, x:x+w] = f
            x += w + 4
        # Add camera label bar above the row
        bar = np.zeros((28, total_w, 3), dtype=np.uint8)
        cv2.putText(bar, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        rows.append(np.vstack([bar, row]))
        print(f"  {label}: {len(frames)} frames from {mp4}")

    grid = np.vstack(rows) if rows else np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(out_path, grid)
    print(f"\nSaved → {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/torque_aug_v0")
    p.add_argument("--out", default="/tmp/cameras_preview.png")
    args = p.parse_args()
    print(f"Dataset: {args.dataset}")
    make_grid(args.dataset, args.out)


if __name__ == "__main__":
    main()
