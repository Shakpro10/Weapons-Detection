import os
import csv
import json
import random
import numpy as np
import torch

from config import (
    # Paths
    WEIGHTS,
    DATA_DIR,
    RESULTS_DIR,
    TRAIN_METADATA_FILE,
    # Training
    RANDOM_SEED,
    BATCH_SIZE,
    EPOCHS,
    RESOLUTION,
    LR_INITIAL,
    LR_ENCODER,
    WEIGHT_DECAY,
    DEVICE,
    PATIENCE,
    EARLY_STOPPING_MIN_DELTA,
    # Model
    NUM_CLASSES,
    # Augmentations
    AUGMENTATION,
)

# RF-DETR — swap class to match MODEL_VARIANT in config.py:
#   RFDETRNano / RFDETRSmall / RFDETRBase / RFDETRMedium / RFDETRLarge / RFDETRXLarge
from rfdetr import RFDETRSmall, RFDETRMedium

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
# Checkpoint helpers
# ==============================

# RF-DETR 1.6.0 checkpoint filenames in priority order for resume.
# "checkpoint_best_regular.pth" is preferred because it contains the full
# training state (model + optimizer + scheduler + epoch) needed to resume.
# "checkpoint_best_ema.pth" is the EMA-smoothed weights-only snapshot.
_CHECKPOINT_CANDIDATES = [
    "last.ckpt",
    "checkpoint_best_regular.pth",   # full training state  ← resume target
    "checkpoint_best_ema.pth",       # EMA weights only     ← fallback
    "checkpoint.pth",                # legacy name (older RF-DETR / manual saves)
]

def _latest_checkpoint(output_dir: str) -> str | None:
    """
    Return the best available checkpoint path inside output_dir, or None.

    RF-DETR 1.6.0 no longer writes 'checkpoint.pth'; it saves:
      checkpoint_best_regular.pth  — full state (preferred for resume)
      checkpoint_best_ema.pth      — EMA weights snapshot
    Older builds / manual saves may still use checkpoint.pth.
    """
    for name in _CHECKPOINT_CANDIDATES:
        candidate = os.path.join(output_dir, name)
        if os.path.exists(candidate):
            return candidate
    return None


def _is_valid_checkpoint(path: str) -> bool:
    """
    Validate that a checkpoint file is a loadable PyTorch state dict.

    RF-DETR 1.6.0 dropped PyTorch Lightning, so checkpoints no longer
    contain 'pytorch-lightning_version'. We now accept any checkpoint that:
      - can be loaded by torch.load without error, AND
      - is a dict (not a raw tensor or something else)

    RF-DETR 1.6.0 full-state checkpoints typically contain:
        'model', 'optimizer', 'lr_scheduler', 'epoch', 'args'
    EMA / weights-only checkpoints contain just:
        'model'  (or the state dict directly)

    Both are accepted — RF-DETR's own resume logic handles the distinction.
    """
    try:
        # weights_only=False is required for PL .ckpt files on PyTorch >= 2.0
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict):
            print(f"   [DEBUG] Checkpoint is not a dict (got {type(ckpt).__name__}) : {path}")
            return False

        # Pytorch-lightning .ckpt files use state dict + 'epoch'/'global_step'
        is_pl_ckpt = "state_dict" in ckpt or "global_step" in ckpt
        # Rfdetr .pth files use 'model' or 'optionally epoch'
        is_rfdetr_pth = "model" in ckpt or "epoch" in ckpt

        if not (is_pl_ckpt or is_rfdetr_pth):
            print(f"   [DEBUG] Unrecognized checkpoint format — keys found: {list(ckpt.keys())}")
            return False

        # Print what was found so you can see it working
        epoch = ckpt.get("epoch", ckpt.get("global_step", "unknown"))
        print(f"   [DEBUG] Valid checkpoint — epoch/step: {epoch} | path: {path}")
        return True

    except Exception as exc:
        print(f"   [DEBUG] Could not load checkpoint ({exc}) : {path}")
        return False


def _resolve_resume(resume_arg: str | None, output_dir: str) -> str | None:
    """
    Resolve the checkpoint to resume from.

    Priority order:
      1. Explicit --resume path passed from CLI (must exist on disk).
      2. Auto-detected checkpoint.pth inside RESULTS_DIR/ (previous session).
      3. None -> fresh training start.

    This means you can simply run:
        python train_rfdetr.py
    and it will automatically resume if a previous checkpoint exists,
    or start fresh if this is the very first run — no flags needed.
    """
    # Priority 1: explicit path provided via CLI
    if resume_arg:
        if os.path.exists(resume_arg):
            # 🔒 Validate checkpoint before using it
            if _is_valid_checkpoint(resume_arg):
                print(f"   Resuming from explicit checkpoint : {resume_arg}")
                return resume_arg
            else:
                print(f"   Provided checkpoint is INVALID    : {resume_arg}")
                print("   Ignoring and falling back to auto-detection ...")
        else:
            print(f"   Explicit --resume path not found  : {resume_arg}")
            print(  "   Falling back to auto-detection ...")

    # Priority 2: auto-detect latest checkpoint from previous session
    auto = _latest_checkpoint(output_dir)
    if auto:
        # 🔒 Validate auto-detected checkpoint before resuming
        if _is_valid_checkpoint(auto):
            print(f"   Checkpoint found — resuming       : {auto}")
            return auto
        else:
            print(f"   Auto-detected checkpoint is INVALID : {auto}")
            print("   Ignoring and falling back to fresh start ...")

    # Priority 3: no checkpoint anywhere -> fresh start
    print("   No checkpoint found — starting fresh training.")
    return None


# ==============================
# Metrics helpers
# ==============================

# Maps each canonical name to all known column name variants across
# RF-DETR versions:
#   - metrics.csv   keys  (RF-DETR >= 1.6 / PyTorch Lightning 2.6.1)
#     Exact column names confirmed from metrics.csv output:
#       val/mAP_50, val/mAP_50_95, val/precision, val/recall
#     EMA variants also present: val/ema_mAP_50, val/ema_mAP_50_95
#   - results.json  keys  (RF-DETR < 1.6, legacy fallback)
_METRIC_ALIASES = {
    "mAP50_95":  (
        # metrics.csv  (RF-DETR 1.6.0) — confirmed column names
        "val/mAP_50_95",    "val/ema_mAP_50_95",
        # results.json (legacy)
        "mAP50_95",        "mAP50-95",         "map50_95",
        "map_50_95",       "map50-95",
    ),
    "mAP50": (
        # metrics.csv  (RF-DETR 1.6.0) — confirmed column names
        "val/mAP_50",       "val/ema_mAP_50",
        # results.json (legacy)
        "mAP50",           "map50",            "mAP_50",
        "map_50",
    ),
    "precision": (
        # metrics.csv  (RF-DETR 1.6.0)
        "val/precision",
        # results.json (legacy)
        "precision",
    ),
    "recall": (
        # metrics.csv  (RF-DETR 1.6.0)
        "val/recall",
        # results.json (legacy)
        "recall",
    ),
    "epoch": (
        "epoch",           # identical in both formats
    ),
}

def _find_key(row: dict, canonical: str) -> str | None:
    """Return the first matching alias key found in row, or None."""
    for alias in _METRIC_ALIASES.get(canonical, ()):
        if alias in row:
            return alias
    return None


# ------------------------------------------------------------------
# RF-DETR >= 1.6 / PyTorch Lightning 2.6.1
# ------------------------------------------------------------------
def _load_metrics_csv(results_dir: str) -> list[dict]:
    """
    Load per-epoch validation metrics from PyTorch Lightning's metrics.csv.

    PyTorch Lightning writes one CSV row per *logging step*, which means
    a single epoch typically produces several rows — some with training
    loss, others with validation metrics — with NaN filling the columns
    that are not applicable to that step.

    This function:
      1. Reads every row from metrics.csv.
      2. Groups rows by epoch number.
      3. Within each group, collapses NaN-scattered values into a single
         dict by taking the last non-NaN value for every column.  This
         reliably captures the final validation metrics logged at the end
         of each epoch regardless of how many training steps preceded them.

    Returns a list of one consolidated dict per epoch, sorted by epoch,
    or an empty list if the file is missing or unreadable.
    """
    path = os.path.join(results_dir, "metrics.csv")
    if not os.path.exists(path):
        print(f"   metrics.csv not found at: {path}")
        return []

    try:
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            raw_rows = list(reader)
    except IOError as exc:
        print(f"   Warning: could not read metrics.csv — {exc}")
        return []

    if not raw_rows:
        print("   metrics.csv is empty.")
        return []

    # ----------------------------------------------------------------
    # Group rows by epoch, then collapse NaNs within each group.
    # ----------------------------------------------------------------
    # epoch_data[epoch_num] = {col: last_non_nan_value, ...}
    epoch_data: dict[int, dict] = {}

    for row in raw_rows:
        # PL writes epoch as a float ("0.0", "1.0", …); coerce to int.
        raw_epoch = row.get("epoch", "").strip()
        try:
            epoch_num = int(float(raw_epoch))
        except (ValueError, TypeError):
            continue  # skip rows with no valid epoch (e.g. header artefacts)

        if epoch_num not in epoch_data:
            epoch_data[epoch_num] = {"epoch": epoch_num}

        for col, val in row.items():
            if col == "epoch":
                continue
            # Keep the last non-NaN, non-empty value seen for this col
            if val is not None and val.strip() not in ("", "nan", "NaN", "NAN"):
                try:
                    epoch_data[epoch_num][col] = float(val)
                except ValueError:
                    epoch_data[epoch_num][col] = val  # keep as string if not numeric

    if not epoch_data:
        print("   metrics.csv contained no parseable epoch rows.")
        return []

    consolidated = [epoch_data[e] for e in sorted(epoch_data)]
    print(f"   Loaded metrics.csv — {len(consolidated)} epoch(s) found.")
    return consolidated


# ------------------------------------------------------------------
# RF-DETR < 1.6 (legacy fallback)
# ------------------------------------------------------------------
def _load_results_json(results_dir: str) -> list[dict]:
    """
    Load RF-DETR's auto-generated results.json (RF-DETR < 1.6 only).

    RF-DETR wrote per-epoch metrics to <output_dir>/results.json in
    older versions.  This function is retained as a fallback so that
    runs started with an older checkpoint can still be resumed and
    their historical metrics parsed correctly.

    Handles both a standard JSON array and newline-delimited JSON
    (one object per line), since RF-DETR's output format varied slightly
    across versions.

    Returns an empty list if the file is missing or unparseable.
    """
    path = os.path.join(results_dir, "results.json")
    if not os.path.exists(path):
        return []

    try:
        with open(path, "r") as fh:
            raw = fh.read().strip()

        # Try standard JSON array first
        if raw.startswith("["):
            data = json.loads(raw)
            if isinstance(data, list):
                print(f"   Loaded results.json (legacy) — {len(data)} epoch(s) found.")
                return data

        # Try newline-delimited JSON (one object per line)
        rows = []
        for line in raw.splitlines():
            line = line.strip().rstrip(",")
            if line:
                rows.append(json.loads(line))
        if rows:
            print(f"   Loaded results.json (legacy, NDJSON) — {len(rows)} epoch(s) found.")
        return rows

    except (json.JSONDecodeError, IOError) as exc:
        print(f"   Warning: could not parse results.json — {exc}")
        return []


def _load_epoch_history(results_dir: str) -> list[dict]:
    """
    Load per-epoch metrics from whichever format is available.

    Priority:
      1. metrics.csv  — RF-DETR >= 1.6 / PyTorch Lightning 2.6.1
      2. results.json — RF-DETR <  1.6 (legacy fallback)

    This allows the same script to be used across both old and new
    RF-DETR versions without manual intervention.
    """
    # Try new format first
    history = _load_metrics_csv(results_dir)
    if history:
        return history

    # Fall back to legacy format
    print("   Falling back to results.json (legacy RF-DETR < 1.6) ...")
    return _load_results_json(results_dir)


# ==============================
# Training
# ==============================
def train_model(resume: str | None = None) -> None:
    """
    Fine-tune RF-DETR on a COCO-format weapons dataset.

    Parameters
    ----------
    resume : str | None
        Explicit path to a checkpoint.pth to resume from.
        Pass None (default) to let the script auto-detect or start fresh.
    """
    set_seeds()

    # -- Effective batch size -------------------------------------------
    # RF-DETR targets effective_batch = 16.
    # grad_accum_steps compensates when BATCH_SIZE < 16 due to VRAM limits.
    #   effective_batch = BATCH_SIZE x GRAD_ACCUM_STEPS
    #   RTX 3060 Ti 8 GB: BATCH_SIZE=2, GRAD_ACCUM=8  -> 2x8=16
    GRAD_ACCUM_STEPS = max(1, 16 // BATCH_SIZE)
    print(f"   Batch: {BATCH_SIZE}  |  Grad accum steps: {GRAD_ACCUM_STEPS}"
          f"  |  Effective batch: {BATCH_SIZE * GRAD_ACCUM_STEPS}")

    # -- Resolve resume checkpoint --------------------------------------
    checkpoint = _resolve_resume(resume, RESULTS_DIR)

    # -- Build model ---------------------------------------------------
    # num_classes MUST match the number of classes in your dataset.
    # Without this, the classification head has the wrong output size.
    if WEIGHTS and os.path.exists(WEIGHTS):
        model = RFDETRMedium(
            pretrain_weights=WEIGHTS,
            num_classes=NUM_CLASSES,
        )
        print(f"   Loaded custom weights   : {WEIGHTS}")
    else:
        model = RFDETRMedium(num_classes=NUM_CLASSES)
        print("   Loaded default RF-DETR-Medium COCO checkpoint")

    import config as _cfg
    print(f"   Classes ({NUM_CLASSES})          : {', '.join(_cfg.CLASSES)}")

    # -- Train ---------------------------------------------------------
    print("\nStarting RF-DETR training ...")
    print("=" * 60)

    try:
        model.train(
            # -- Dataset -----------------------------------------------
            # RF-DETR auto-detects YOLO format.
            # Expected YOLO structure:
            #   DATA_DIR/
            #     train/  images/  labels/
            #     valid/  images/  labels/
            #     test/   images/  labels/
            #     data.yaml
            dataset_dir=DATA_DIR,

            # -- Core --------------------------------------------------
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            grad_accum_steps=GRAD_ACCUM_STEPS,
            output_dir=RESULTS_DIR,
            progress_bar=True,  # show tqdm progress bar during training

            # -- Resolution --------------------------------------------
            # Must be divisible by 56.
            # 672 recommended for CCTV deployment on 8 GB VRAM.
            resolution=RESOLUTION,

            # -- Learning rates ----------------------------------------
            # lr        -> main decoder/head learning rate
            # lr_encoder -> backbone encoder LR
            lr=LR_INITIAL,
            lr_encoder=LR_ENCODER,

            # -- Regularisation ----------------------------------------
            weight_decay=WEIGHT_DECAY,

            # -- Hardware ----------------------------------------------
            device=DEVICE,
            # gradient_checkpointing: re-computes forward pass during
            # backprop to save VRAM at ~20% slower training speed.
            # Set True if you get OOM errors on 8 GB VRAM.
            gradient_checkpointing=False,

            # -- Speed optimisations -----------------------------------
            # num_workers: parallel CPU workers that pre-fetch batches while
            # the GPU is busy training. 0 = single-threaded (slow).
            # With 32 GB RAM, 4 workers is a safe default.
            # Increase to 6-8 if your CPU has enough cores and you see
            # GPU utilisation dipping below 80% between batches.
            num_workers=10,
            persistent_workers=True,

            # pin_memory: pre-allocates page-locked CPU memory so each
            # batch transfer to the GPU is ~20-30% faster. Always True
            # when training on CUDA.
            pin_memory=True,

            # Each worker pre-fetches this many batches in the background. 
            # 2-4 is a good default; increase if you have enough RAM and see 
            # GPU utilisation dipping below 80% between batches. Set to 0 to 
            # disable pre-fetching (not recommended).
            prefetch_factor=4,

            # use_amp: Automatic Mixed Precision (FP16 and FP32 on GPU).
            # Halves VRAM usage for activations and speeds up compute on
            # the RTX 3060 Ti's Tensor Cores by roughly 1.5-2x.
            # Negligible accuracy impact for detection tasks.
            precision="16-mixed",
            # use_amp=True,

            # -- EMA ---------------------------------------------------
            # Exponential Moving Average smooths the final model.
            # Usually improves mAP by 0.5-1.5 points. Keep True.
            use_ema=True,

            # -- Checkpointing -----------------------------------------
            # Saves checkpoint_<N>.pth every N epochs in addition to:
            #   checkpoint.pth            -> latest (resume target)
            #   checkpoint_best_total.pth -> best mAP epoch
            checkpoint_interval=5,

            # -- Early stopping ----------------------------------------
            early_stopping=True,
            early_stopping_patience=PATIENCE,
            early_stopping_min_delta=EARLY_STOPPING_MIN_DELTA,
            early_stopping_use_ema=True,

            # -- Resume ------------------------------------------------
            # Restores model weights, optimizer state, AND lr scheduler.
            # None = fresh start.
            resume=checkpoint,

            # -- Augmentations -----------------------------------------
            # Albumentations dict API.
            # RF-DETR applies transforms to images AND bboxes automatically.
            aug_config=AUGMENTATION,
            bbox_params=dict(
                format='yolo',           # or 'coco' if your labels are in COCO format
                label_fields=['labels'], # RF-DETR expects your dict to include 'labels'
                min_visibility=0.0       # optional; prevents fully invisible boxes from being used
            ),
        )

        print("\nTraining completed successfully.")

    except Exception as exc:
        print(f"\nTraining failed: {exc}")
        raise

    # -- Save best-epoch metrics (reads metrics.csv or results.json) ---
    _save_metrics(RESULTS_DIR)


# ==============================
# Metrics persistence
# ==============================
def _save_metrics(results_dir: str) -> None:
    """
    Parse RF-DETR's per-epoch metrics to find the best epoch by
    mAP50-95, compute F1, and write train_metadata.json.

    Source priority:
      1. metrics.csv  — RF-DETR >= 1.6 / PyTorch Lightning 2.6.1
         PL writes one row per logging step; _load_metrics_csv()
         consolidates these into one clean dict per epoch.
      2. results.json — RF-DETR < 1.6 (legacy fallback)
         Retained so that runs from older versions are still handled
         correctly without any manual changes.
    """
    history = _load_epoch_history(results_dir)

    if not history:
        print("No epoch data found in metrics.csv or results.json — skipping metadata save.")
        return

    # Identify which key names RF-DETR / PL used in this particular run
    map_key   = _find_key(history[-1], "mAP50_95")
    map50_key = _find_key(history[-1], "mAP50")
    prec_key  = _find_key(history[-1], "precision")
    rec_key   = _find_key(history[-1], "recall")
    epoch_key = _find_key(history[-1], "epoch")

    if map_key is None:
        print("   Could not find mAP50-95 key in epoch history.")
        print(f"   Available keys: {list(history[-1].keys())}")
        print("   Falling back to last epoch for metadata.")
        best_row = history[-1]
    else:
        # Pick epoch with highest mAP50-95 — same logic as YOLO's idxmax()
        best_row = max(history, key=lambda r: float(r.get(map_key, 0.0)))

    precision = float(best_row[prec_key])  if prec_key  and prec_key  in best_row else None
    recall    = float(best_row[rec_key])   if rec_key   and rec_key   in best_row else None
    map50     = float(best_row[map50_key]) if map50_key and map50_key in best_row else None
    map50_95  = float(best_row[map_key])   if map_key   and map_key   in best_row else None
    epoch     = int(best_row[epoch_key])   if epoch_key and epoch_key in best_row else None

    # Compute F1 — mirrors YOLO script's fallback formula exactly
    f1_score = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1_score = 2 * precision * recall / (precision + recall)

    train_subdir = results_dir  # RF-DETR saves checkpoints directly in RESULTS_DIR

    metrics = {
        "epoch":        epoch,
        "precision":    round(precision, 6) if precision is not None else None,
        "recall":       round(recall,    6) if recall    is not None else None,
        "f1_score":     round(f1_score,  6) if f1_score  is not None else None,
        "mAP50":        round(map50,     6) if map50     is not None else None,
        "mAP50-95":     round(map50_95,  6) if map50_95  is not None else None,
        "best_weights": os.path.join(train_subdir, "checkpoint_best_total.pth"),
        "regular_weight": os.path.join(train_subdir, "checkpoint_best_regular.pth"),
        "ema_weights":  os.path.join(train_subdir, "checkpoint_best_ema.pth"),
        "last_weights": os.path.join(train_subdir, "last.ckpt"),
    }

    os.makedirs(os.path.dirname(TRAIN_METADATA_FILE), exist_ok=True)
    with open(TRAIN_METADATA_FILE, "w") as fh:
        json.dump(metrics, fh, indent=4)

    print(f"\nBest epoch metrics saved -> {TRAIN_METADATA_FILE}")
    print(f"\n{'='*60}")
    print(f"  Best Model Performance")
    print(f"{'='*60}")
    print(f"  Epoch      : {metrics['epoch']}")
    print(f"  Precision  : {metrics['precision']:.4f}"  if metrics["precision"]  is not None else "  Precision  : N/A")
    print(f"  Recall     : {metrics['recall']:.4f}"     if metrics["recall"]     is not None else "  Recall     : N/A")
    print(f"  F1 Score   : {metrics['f1_score']:.4f}"   if metrics["f1_score"]   is not None else "  F1 Score   : N/A")
    print(f"  mAP@0.5    : {metrics['mAP50']:.4f}"      if metrics["mAP50"]      is not None else "  mAP@0.5    : N/A")
    print(f"  mAP@0.5:95 : {metrics['mAP50-95']:.4f}"   if metrics["mAP50-95"]   is not None else "  mAP@0.5:95 : N/A")
    print(f"{'='*60}")
    print(f"\n  Training artifacts saved to : {train_subdir}")
    print(f"  Best weights  : {metrics['best_weights']}")
    print(f"  EMA weights   : {metrics['ema_weights']}")
    print(f"  Periodic saves: {train_subdir}/checkpoint_<N>.pth  (every 5 epochs)")


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    import argparse
    import config as _cfg

    parser = argparse.ArgumentParser(
        description="RF-DETR Weapon Detection Training Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="CHECKPOINT",
        help=(
            "Path to a checkpoint.pth to resume training from.\n"
            "\n"
            "Behaviour:\n"
            "  --resume path/to/checkpoint.pth  -> resume from that exact file\n"
            "  (no flag, checkpoint exists)      -> auto-resume from last session\n"
            "  (no flag, no checkpoint on disk)  -> start fresh training\n"
        ),
    )
    args = parser.parse_args()

    # Startup banner
    print("=" * 60)
    print("  RF-DETR Weapon Detection — Training Pipeline")
    print("=" * 60)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_vram = torch.cuda.get_device_properties(0).total_memory // 1024**3
        print(f"  Device     : {gpu_name}")
        print(f"  VRAM       : {gpu_vram} GB")
    else:
        print("  Device     : CPU (no CUDA GPU detected)")
    print(f"  Model      : {_cfg.MODEL_VARIANT}  |  Classes: {NUM_CLASSES}")
    print(f"  Resolution : {_cfg.RESOLUTION}")
    print(f"  Batch size : {BATCH_SIZE}  (effective: {BATCH_SIZE * max(1, 16 // BATCH_SIZE)})")
    print(f"  Epochs     : {EPOCHS}")
    print(f"  Dataset    : {DATA_DIR}")
    print(f"  Output     : {RESULTS_DIR}")
    print("=" * 60)

    train_model(resume=args.resume)

    print("=" * 60)
    print("  Pipeline finished!")
    print("=" * 60)