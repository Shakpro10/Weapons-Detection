import os
import glob
import math
import random

import cv2
import numpy as np
import torch
import supervision as sv
from PIL import Image
from rfdetr import RFDETRSmall

from config import RANDOM_SEED, RESOLUTION, DEVICE

# ==============================
# Class definitions with colors
# ==============================
# Colors in BGR format (OpenCV convention)
CLASSES = {
    0: {"name": "Handgun",                  "color": (128,   0, 128)},  # Purple
    1: {"name": "Heavy Weapon",             "color": (255,   0,   0)},  # Blue
    2: {"name": "Knife",                    "color": (  0, 255,   0)},  # Green
    3: {"name": "Rifle",                    "color": (  0,   0, 255)},  # Red
    4: {"name": "Rocket_Grenade Launcher",  "color": (  0, 165, 255)},  # Orange
}

NUM_CLASSES = len(CLASSES)

# Supervision ColorPalette built from our class colors (expects RGB tuples)
PALETTE = sv.ColorPalette(colors=[
    sv.Color(r=128, g=0,   b=128),  # Handgun        — Purple
    sv.Color(r=0,   g=0,   b=255),  # Heavy Weapon   — Blue
    sv.Color(r=0,   g=255, b=0  ),  # Knife          — Green
    sv.Color(r=255, g=0,   b=0  ),  # Rifle          — Red
    sv.Color(r=255, g=165, b=0  ),  # Rocket Grenade — Orange
])

# ==============================
# Reproducibility
# ==============================
def set_seeds(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================
# Load RF-DETR model
# ==============================
BEST_WEIGHTS = r"runs/checkpoint_best_total.pth"

print("=" * 60)
print("  Loading RF-DETR-Medium model …")
print(f"  Weights : {BEST_WEIGHTS}")
print(f"  Device  : {DEVICE}")
print("=" * 60)

model = RFDETRSmall(
    pretrain_weights=BEST_WEIGHTS,
    num_classes=NUM_CLASSES,
)

# Supervision annotators — used instead of cvzone
box_annotator   = sv.BoxAnnotator(color=PALETTE, thickness=2)
label_annotator = sv.LabelAnnotator(
    color=PALETTE,
    text_color=sv.Color.WHITE,
    text_scale=0.6,
    text_thickness=2,
    text_padding=4,
)

WINDOW_NAME = "RF-DETR Weapon Detection"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.moveWindow(WINDOW_NAME, 200, 80)

exit_all    = False
frame_step  = 30  # frames to skip per fast-forward / rewind keypress


# ==============================
# Run detection on video
# ==============================
def run_video_inference(video_source=None, conf_thres: float = 0.35) -> None:
    """
    Run RF-DETR weapon detection on one or more video files, or webcam.

    Parameters
    ----------
    video_source : None | str | list[str]
        None          → default webcam (index 0)
        str           → single video file path
        list[str]     → multiple video file paths
    conf_thres : float
        Confidence threshold — detections below this are ignored.
        RF-DETR is more conservative than YOLO; 0.35 is a good starting point.
    """
    global exit_all
    set_seeds()

    # ── Build list of capture sources ─────────────────────────────────
    if video_source is None:
        print("  Starting webcam stream …")
        sources      = [cv2.VideoCapture(0)]
        source_names = ["Webcam"]
    elif isinstance(video_source, (list, tuple)):
        sources      = [cv2.VideoCapture(vp) for vp in video_source]
        source_names = [os.path.basename(vp) for vp in video_source]
    elif isinstance(video_source, str) and os.path.exists(video_source):
        sources      = [cv2.VideoCapture(video_source)]
        source_names = [os.path.basename(video_source)]
    else:
        print("  Error: Invalid video source(s).")
        return

    # ── Process each video ─────────────────────────────────────────────
    for cap, video_name in zip(sources, source_names):
        if not cap.isOpened():
            print(f"  Error: Could not open — {video_name}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS)
        delay        = int(1000 / fps) if fps > 0 else 1
        frame_index  = 0

        print("=" * 60)
        print(f"  Processing : {video_name}")
        print(f"  Frames     : {total_frames}  |  FPS: {fps:.2f}")
        print("=" * 60)

        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame_bgr = cap.read()
            if not success:
                break   # end of video

            # ── RF-DETR requires RGB — convert from OpenCV's BGR ──────
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # ── Run inference ─────────────────────────────────────────
            # model.predict() accepts a PIL Image or numpy RGB array.
            # It returns a supervision.Detections object directly —
            # no NMS needed (RF-DETR is end-to-end by design).
            detections = model.predict(frame_rgb, threshold=conf_thres)

            # ── Build labels for each detection ───────────────────────
            labels = []
            for class_id, confidence in zip(detections.class_id, detections.confidence):
                class_info = CLASSES.get(int(class_id), {"name": f"Class_{class_id}"})
                label = f"{class_info['name']} {confidence:.2f}"
                labels.append(label)
                print(f"  {class_info['name']} detected — confidence: {confidence:.2f}")

            # ── Draw bounding boxes and labels using supervision ──────
            # supervision annotators work on BGR frames directly
            annotated_frame = frame_bgr.copy()
            annotated_frame = box_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
            )
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                labels=labels,
            )

            # ── Overlay video info ────────────────────────────────────
            info_text = f"Video: {video_name}  |  Frame: {frame_index}/{total_frames}"
            fps_text  = f"FPS: {fps:.1f}  |  RF-DETR (End-to-End, No NMS)  |  Conf: {conf_thres}"
            cv2.putText(annotated_frame, info_text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255,   0), 2)
            cv2.putText(annotated_frame, fps_text,  (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (  0, 255, 255), 2)

            cv2.imshow(WINDOW_NAME, annotated_frame)

            # ── Keyboard controls (identical to YOLO script) ──────────
            key = cv2.waitKey(delay) & 0xFF
            if key == ord('q'):
                print("  Skipping to next video …")
                break
            elif key == ord('e'):
                print("  Exiting all videos …")
                exit_all = True
                break
            elif key == ord('d'):   # fast-forward
                frame_index = min(frame_index + frame_step, total_frames - 1)
                continue
            elif key == ord('a'):   # rewind
                frame_index = max(frame_index - frame_step, 0)
                continue
            else:
                frame_index += 1

        cap.release()
        print(f"  Finished: {video_name}")

        if exit_all:
            break

    cv2.destroyAllWindows()
    print("  All videos processed.")


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    print("=" * 60)
    print("  RF-DETR Weapon Detection — Video Inference")
    print("=" * 60)
    print("  Controls:")
    print("    Q — Skip to next video")
    print("    E — Exit all videos")
    print("    D — Fast forward 30 frames")
    print("    A — Rewind 30 frames")
    print("=" * 60)

    # ── Video directory ────────────────────────────────────────────────
    video_dir = r"C:\Users\Shakiru\ML\Weapon_Detection\v26\test_videos"

    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv',
                        '*.MP4', '*.AVI', '*.MOV', '*.MKV']
    video_paths = []
    for ext in video_extensions:
        video_paths.extend(glob.glob(os.path.join(video_dir, ext)))

    if not video_paths:
        print(f"  No video files found in: {video_dir}")
        print("  Starting webcam instead …")
        run_video_inference(None, conf_thres=0.75)
    else:
        print(f"  Found {len(video_paths)} video(s):")
        for i, vp in enumerate(video_paths, 1):
            print(f"    {i}. {os.path.basename(vp)}")
        print("=" * 60)
        run_video_inference(video_paths, conf_thres=0.75)

    print("=" * 60)
    print("  Video inference completed.")
    print("=" * 60)