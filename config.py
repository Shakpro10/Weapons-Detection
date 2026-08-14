import os
import torch

# ==========================
# CONFIGURATION SETTINGS
# ==========================

# --- PATHS & DATASET ---
BASE_DIR = os.getcwd()

# RF-DETR auto-detects YOLO format — no conversion needed.
# Expected YOLO structure (YOLOv8 export from Roboflow):
#   DATA_DIR/
#     train/  images/  labels/
#     valid/  images/  labels/
#     test/   images/  labels/
#     data.yaml
DATA_DIR = os.path.join(BASE_DIR, "dataset", "Weapons_Detection_New_v2_version_2_yolo26")
print(DATA_DIR)

# --- OUTPUT & WEIGHTS ---
# RF-DETR saves checkpoints under RESULTS_DIR/train/
RESULTS_DIR = os.path.join(BASE_DIR, "runs")

# Checkpoint paths written by RF-DETR automatically:
#   checkpoint.pth              → latest  (used for --resume)
#   checkpoint_best_total.pth   → best mAP epoch
#   checkpoint_<N>.pth          → periodic saves every checkpoint_interval epochs
BEST_WEIGHTS  = os.path.join(RESULTS_DIR, "checkpoint_best_total.pth")
REGULAR_WEIGHTS = os.path.join(RESULTS_DIR, "checkpoint_best_regular.pth") 
EMA_WEIGHTS = os.path.join(RESULTS_DIR, "checkpoint_best_ema.pth") 
LAST_WEIGHTS  = os.path.join(RESULTS_DIR, "last.ckpt")

# Pretrained / custom starting weights
# Set to None to download the default RF-DETR-Base COCO checkpoint automatically.
# Set to a local .pth path to start from your own checkpoint.
WEIGHTS = None  # e.g. "path/to/my_pretrained.pth"

# --- METADATA FILES ---
TRAIN_METADATA_FILE = os.path.join(RESULTS_DIR, "train_metadata.json")
TEST_METADATA_FILE  = os.path.join(RESULTS_DIR, "test_metadata.json")

# ==========================
# TRAINING PARAMETERS
# ==========================

EPOCHS      = 100    # Total training epochs
BATCH_SIZE  = 8      # Per-GPU batch size — effective batch = BATCH_SIZE × grad_accum_steps
                     # RF-DETR targets effective batch of 16; train.py auto-sets grad_accum_steps
                     # GPU VRAM guide:
                     #   A100 (40-80 GB) → BATCH_SIZE = 16  (grad_accum = 1)
                     #   RTX 4090/3090   → BATCH_SIZE = 8   (grad_accum = 2)
                     #   T4 (16 GB)      → BATCH_SIZE = 4   (grad_accum = 4)
                     #   RTX 3070 (8 GB) → BATCH_SIZE = 2   (grad_accum = 8)

# Resolution — must be divisible by 56
# 560 → low memory   672 → balanced (default)   784 → high accuracy   896 → max quality
RESOLUTION = 672

# --- LEARNING RATES ---
# RF-DETR uses separate LRs for the backbone encoder vs the rest of the model.
# LR_ENCODER is typically set ≥ LR_INITIAL (RF-DETR default ratio is 1.5×).
LR_INITIAL  = 1e-4           # Main model learning rate
LR_ENCODER  = 1.5e-4        # Backbone encoder learning rate (set lower to freeze encoder gently)

# --- REGULARISATION ---
WEIGHT_DECAY = 1e-4          # L2 penalty; helps prevent overfitting

# --- EARLY STOPPING ---
PATIENCE                 = 10       # Epochs without improvement before stopping
EARLY_STOPPING_MIN_DELTA = 0.005    # Minimum mAP improvement to count as progress

# --- DEVICE ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# For a specific GPU: DEVICE = "cuda:0"
# For Apple Silicon:  DEVICE = "mps"

# ==========================
# REPRODUCIBILITY
# ==========================
RANDOM_SEED = 42

# ==========================
# DETECTION CLASSES
# ==========================
CLASSES = [
    "Handgun",
    "Heavy Weapon",
    "Knife",
    "Rifle",
    "Rocket_Grenade Launcher",
]
NUM_CLASSES = len(CLASSES)

# ==========================
# RF-DETR MODEL VARIANT
# ==========================
# Choose based on your accuracy/speed trade-off:
#   RFDETRNano    → fastest, least accurate
#   RFDETRSmall   → good balance for edge deployment
#   RFDETRBase    → recommended default  ← used in train.py
#   RFDETRMedium  → higher accuracy, more VRAM
#   RFDETRLarge   → best accuracy, requires ≥24 GB VRAM
#   RFDETRXLarge  → research/maximum quality
MODEL_VARIANT = "RFDETRSmall"  # change here and import accordingly in train.py

# ==============================
# Augmentation config (Albumentations dict API)
# RF-DETR automatically handles bbox transformation for all spatial ops.
# Keys are Albumentations class names; values are their kwargs + "p".
# ==============================

AUGMENTATION = {

    # =========================================================================
    # COLOUR / PHOTOMETRIC
    # =========================================================================

    # CCTV cameras operate across wildly different lighting conditions:
    # bright outdoor daylight, dim indoor fluorescent, overexposed entranceways,
    # underexposed carparks. This is the single most important photometric
    # augmentation for CCTV — set HIGH.
    # Handguns and knives are small; brightness/contrast shifts are the #1
    # reason a small weapon becomes invisible to a model trained only on
    # clean, well-lit images.
    "RandomBrightnessContrast": {
        "brightness_limit": 0.35,
        "contrast_limit":   0.35,
        "p": 0.70,              # HIGH — lighting variation is universal in CCTV
    },

    # HSV shifts — covers YOLO's hsv_h=0.015, hsv_s=0.5, hsv_v=0.4
    # Covers hue, saturation and value (brightness) shifts simultaneously.
    # Different CCTV brands produce noticeably different colour outputs.
    # IR-cut filters change colour rendering at dusk. Important but slightly
    # lower than brightness/contrast because saturation matters less than
    # luminance for detection tasks.
    "HueSaturationValue": {
        "hue_shift_limit": 15,      # hsv_h equivalent
        "sat_shift_limit": 50,      # hsv_s equivalent (raised to match YOLO's 0.5)
        "val_shift_limit": 40,      # hsv_v equivalent (raised to match YOLO's 0.4)
        "p": 0.50,                  # MEDIUM-HIGH — camera brand/IR-cut variation
    },

    # Simulates the warm/cool colour cast differences between camera models
    # and different artificial light sources (sodium vs LED vs fluorescent).
    # Less critical than HSV but adds useful diversity.
    "RGBShift": {
        "r_shift_limit": 15,
        "g_shift_limit": 15,
        "b_shift_limit": 15,
        "p": 0.25,                  # MEDIUM — camera brand colour cast
    },

    # Enhances local contrast — simulates what happens in low-light CCTV
    # scenes where the camera's AGC (auto gain control) kicks in.
    # Also helps model learn features in washed-out overexposed areas
    # (e.g. bright doorways where a weapon might be carried).
    "CLAHE": {
        "clip_limit":     4.0,
        "tile_grid_size": [8, 8],
        "p": 0.25,
    },

    # Sharpening simulates the artificial edge-enhancement many CCTV DVRs
    # apply during recording or streaming. Also compensates for the slight
    # softness introduced by Downscale augmentation elsewhere in the pipeline.
    # Knives especially benefit — their thin blade profile depends on sharp edges.
    "Sharpen": {
        "alpha": [0.2, 0.5],
        "lightness": [0.75, 1.0],
        "p": 0.2,                   # MEDIUM — DVR edge-enhancement + knife edge detail
    },

    # Simulates night-vision / IR-mode CCTV which produces greyscale output.
    # Most modern CCTV cameras switch to IR greyscale in low-light automatically.
    # Keep this LOW — daytime colour footage is still more common than IR mode,
    # but the model must not fail completely when it encounters greyscale feeds.
    "ToGray": {"p": 0.12},           # LOW-MEDIUM — IR night vision mode simulation
          

    # =========================================================================
    # BLUR / NOISE / COMPRESSION
    # =========================================================================

    # Lens blur from cheap CCTV optics — very common in budget installations.
    # Also simulates slight defocus when the camera's autofocus hunts.
    # Critical for Handgun and Knife classes whose small size makes them
    # the first to disappear under blur.
    "GaussianBlur": {
        "blur_limit": [3, 7],
        "p": 0.20,              # MEDIUM — cheap lens / defocus simulation
    },
    
    # Motion blur from a person walking/running with a weapon, or from
    # camera vibration (outdoor poles, fans, vibration). Heavier weapons
    # like Rifles and Heavy Weapons cause more motion blur due to their
    # length and the arm swing involved in carrying them.
    "MotionBlur": {
        "blur_limit": [3, 9],
        "p": 0.25,              # MEDIUM — movement blur very common in CCTV
    },

    # Sensor noise from low-light gain amplification. Budget CCTV cameras
    # produce significant noise after dark. This is especially damaging to
    # Handgun and Knife detection since their small features are buried in noise.
    "GaussNoise": {
        # 1.4.x uses var_limit (variance), not std_range (std deviation).
        # Original std range [0.01, 0.06] → var = std² → [0.0001, 0.0036]
        # Albumentations internally expects var_limit as pixel-scale (0-255):
        # multiply by 255² → approx [6.5, 233] — use rounded safe range:
        "std_range": [0.02, 0.1],
        "p": 0.25,              # MEDIUM — low-light sensor noise
    },

    # THE most important compression augmentation for CCTV.
    # All CCTV systems compress footage with H.264/H.265 at aggressive bitrates
    # to save storage. Compression creates blocking artefacts that are permanent
    # in stored footage. This is not optional for CCTV deployment —
    # set HIGH. Knives are particularly vulnerable as their thin shape
    # gets destroyed by DCT block artefacts.
    "ImageCompression": {
        "quality_range": [50, 90],  # floor lowered to 50 — cheap DVRs compress hard
        "p": 0.45,              # HIGH — CCTV compression is universal
    },

    # =========================================================================
    # GEOMETRIC / LENS DISTORTION
    # =========================================================================

    # Fisheye / barrel / pincushion distortion.
    # Important for fisheye CCTV, dome cameras and wide-angle lenses.
    #
    # NOTE:
    # Do NOT use this aggressively. Excessive distortion can make weapons
    # unrealistically deformed and hurt normal-camera performance.
    "OpticalDistortion": {
        "distort_limit": (-0.20, 0.20),
        "p": 0.20,
    },


    # =========================================================================
    # SMALL OBJECT / DISTANT WEAPON AUGMENTATION
    # =========================================================================

    # Simulates weapons appearing at different distances and sizes.
    # Your existing Affine already provides scale variation, so this should
    # complement it rather than replace it.
    "RandomScale": {
        "scale_limit": (-0.30, 0.30),
        "p": 0.25,
    },

    # Simulates digital zoom (PTZ cameras zooming in) and low-resolution
    # cameras being upscaled by the DVR monitor for display. Very common
    # in older CCTV installations still in service.
    "Downscale": {
        "scale_range": [0.5, 0.9],
        "p": 0.25,              # MEDIUM — older cameras / digital zoom
    },

    # =========================================================================
    # GEOMETRIC / SPATIAL
    # =========================================================================

    # Weapons are carried on both sides of the body, held in either hand,
    # and CCTV cameras capture people approaching from all directions.
    # A Handgun in a right hand looks different to one in a left hand.
    # Set MEDIUM-HIGH — not as high as brightness because geometric
    # flip doubles your effective dataset for left/right asymmetry.
    "HorizontalFlip": {
        "p": 0.50,              # MEDIUM-HIGH — left/right weapon carry variation
    },

    # Combined shift + scale + rotate + shear in one transform (Albumentations 2.0).
    # This is the most important geometric augmentation overall:
    # - scale: weapons appear at very different sizes in CCTV (near vs far)
    # - rotate: people lean, crouch, or the camera is mounted at an angle
    # - translate: weapon is not always centred in the frame
    # - shear: lens distortion at frame edges
    # Set HIGH — scale and position variation are the #1 geometric challenge
    # for weapon detection in CCTV where subjects move freely in the frame.
    "Affine": {
        "translate_percent": {"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        "scale":             (0.6, 1.4),
        "rotate":            (-15, 15),
        "shear":             {"x": (-3, 3), "y": (-3, 3)},
        "fill":              0,
        "p": 0.60,              # HIGH — scale/position variation critical for CCTV
    },

    # CCTV cameras are mounted at angles — ceiling, wall brackets, poles.
    # Perspective distortion is not subtle; subjects far from the camera
    # look very different to subjects close to it in the same frame.
    # Important for Rifle and Heavy Weapon which have distinctive shapes
    # that perspective distortion changes dramatically.
    "Perspective": {
        "scale": [0.02, 0.07],
        "p": 0.30,              # MEDIUM — angled CCTV mounting positions
    },

    # =========================================================================
    # CCTV-SPECIFIC ENVIRONMENTAL
    # =========================================================================

    # Rain on outdoor dome covers. Important for outdoor installations but
    # indoor cameras never see this. Set LOW — only a fraction of deployments
    # are outdoor and exposed, but when they are, rain is a real challenge.
    "RandomRain": {
        "slant_range":            [-10, 10],
        "drop_length":            20,
        "drop_width":             1,
        "drop_color":             [200, 200, 200],
        "blur_value":             2,
        "brightness_coefficient": 0.9,
        "rain_type":              "drizzle",
        "p": 0.08,              # LOW — outdoor cameras only, but must be handled
    },

    # Fog/mist — outdoor cameras in humid climates or near water.
    # Lagos (your location) has a humid climate with harmattan haze.
    # Keep LOW but present — haze reduces contrast and washes out
    # small weapons at distance.
    "RandomFog": {
        "fog_coef_range": [0.1, 0.3],
        "alpha_coef":     0.08,
        "p": 0.10,              # LOW — harmattan haze / humid outdoor cameras
    },

    # Shadows are EXTREMELY common in CCTV — sunlight through windows,
    # ceiling light patterns, doorframe shadows, tree shadows outdoors.
    # A shadow falling across a weapon (especially a dark handgun or knife)
    # can make it near-invisible. Set MEDIUM-HIGH.
    "RandomShadow": {
        "shadow_roi":        [0.0, 0.0, 1.0, 1.0],  # full frame — shadows anywhere
        "num_shadows_limit": [1, 3],                # up to 3 shadow regions
        "shadow_dimension":  5,
        "p": 0.35,              # MEDIUM-HIGH — shadows universal in fixed CCTV
    },

    # =========================================================================
    # OCCLUSION / DROPOUT
    # =========================================================================

    # THE most important augmentation for weapon detection specifically.
    # Weapons are routinely partially concealed:
    # - Handguns tucked into waistbands (only grip visible)
    # - Knives partially behind clothing or held low
    # - Rifles slung across back (partially behind body)
    # Coarse dropout forces the model to detect weapons from partial views
    # rather than relying on seeing the complete weapon shape.
    # Set MEDIUM-HIGH — partial occlusion is the #1 real-world challenge
    # for weapon detection that training datasets underrepresent.
    "CoarseDropout": {
        "num_holes_range":   [1, 8],        # up to 8 occlusion patches
        "hole_height_range": [13, 101],     # up to 15% of image height
        "hole_width_range":  [13, 101],     # up to 15% of image width
        "fill":              0,
        "p": 0.30,              # MEDIUM-HIGH — partial concealment is universal
    },

    
}