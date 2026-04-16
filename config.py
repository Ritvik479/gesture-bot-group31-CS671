# config.py
# Central configuration for the gesture-bot pipeline.
# Every other file imports from here. Nothing is hardcoded elsewhere.

# ─────────────────────────────────────────────
# GESTURE DEFINITIONS
# ─────────────────────────────────────────────

# Gesture class names — order defines the integer label used during training.
# Label 0 is always IDLE. Do not reorder without retraining the model.
GESTURE_CLASSES = [
    "IDLE",           # 0 — no gesture / resting state
    "MOVE_FORWARD",   # 1 — right arm raised overhead
    "MOVE_BACKWARD",  # 2 — both arms raised overhead
    "STOP",           # 3 — T-pose, both arms extended sideways
    "TURN_LEFT",      # 4 — left arm pointing horizontally to the left
    "TURN_RIGHT",     # 5 — right arm pointing horizontally to the right
]

NUM_CLASSES = len(GESTURE_CLASSES)

# Maps gesture class name → robot action string sent to sim_controller
GESTURE_TO_ACTION = {
    "IDLE":          "HOLD",
    "MOVE_FORWARD":  "MOVE_FORWARD",
    "MOVE_BACKWARD": "MOVE_BACKWARD",
    "STOP":          "STOP",
    "TURN_LEFT":     "TURN_LEFT",
    "TURN_RIGHT":    "TURN_RIGHT",
}

# ─────────────────────────────────────────────
# MEDIAPIPE LANDMARK INDICES
# ─────────────────────────────────────────────
# Named indices into the 33-landmark MediaPipe Pose output.
# Full map: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker

LANDMARK = {
    "NOSE":             0,
    "LEFT_SHOULDER":    11,
    "RIGHT_SHOULDER":   12,
    "LEFT_ELBOW":       13,
    "RIGHT_ELBOW":      14,
    "LEFT_WRIST":       15,
    "RIGHT_WRIST":      16,
    "LEFT_HIP":         23,
    "RIGHT_HIP":        24,
    "LEFT_KNEE":        25,
    "RIGHT_KNEE":       26,
    "LEFT_ANKLE":       27,
    "RIGHT_ANKLE":      28,
}

# ─────────────────────────────────────────────
# SLIDING WINDOW / SEQUENCE SETTINGS
# ─────────────────────────────────────────────

# Number of frames per gesture sequence fed into the LSTM.
# At 30fps this equals 1 second of context.
SEQUENCE_LENGTH = 30

# MediaPipe outputs 33 landmarks × 3 coordinates (x, y, z) = 99 values per frame.
NUM_LANDMARKS = 33
COORDS_PER_LANDMARK = 3
FEATURE_DIM = NUM_LANDMARKS * COORDS_PER_LANDMARK  # 99

# ─────────────────────────────────────────────
# INFERENCE / CONFIDENCE SETTINGS
# ─────────────────────────────────────────────

# Minimum softmax confidence to accept a gesture prediction.
# Below this threshold the output is treated as IDLE.
CONFIDENCE_THRESHOLD = 0.85

# Number of consecutive frames a gesture must be held before a command fires.
# At 30fps, 15 frames = 0.5 seconds. Prevents flickering commands.
DEBOUNCE_FRAMES = 15

# ─────────────────────────────────────────────
# WEBCAM SETTINGS
# ─────────────────────────────────────────────

CAMERA_INDEX = 0          # 0 = default webcam
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
TARGET_FPS   = 30

# ─────────────────────────────────────────────
# DATA COLLECTION SETTINGS
# ─────────────────────────────────────────────

# Countdown (in seconds) shown before each capture begins.
COLLECTION_COUNTDOWN_SEC = 3

# Number of samples to collect per session before the script reminds
# you to take a break / switch person. Not a hard limit.
SAMPLES_PER_SESSION_REMINDER = 50

# Key bindings for the data collection script (keyboard key → class index).
# Keys 1–5 map to gestures; key 0 captures IDLE frames.
COLLECTION_KEYBINDINGS = {
    "0": 0,   # IDLE
    "1": 1,   # MOVE_FORWARD
    "2": 2,   # MOVE_BACKWARD
    "3": 3,   # STOP
    "4": 4,   # TURN_LEFT
    "5": 5,   # TURN_RIGHT
}

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

import os

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))

DATA_RAW_DIR    = os.path.join(BASE_DIR, "data", "raw")
DATA_PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR      = os.path.join(BASE_DIR, "models")
LOGS_DIR        = os.path.join(BASE_DIR, "logs")

MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "best_model.keras")
METADATA_CSV    = os.path.join(DATA_RAW_DIR, "metadata.csv")

MUJOCO_MODEL_PATH = os.path.join(BASE_DIR, "simulation", "humanoid.xml")

# ─────────────────────────────────────────────
# LSTM MODEL HYPERPARAMETERS
# ─────────────────────────────────────────────

LSTM_UNITS      = 64      # hidden units per LSTM layer
LSTM_LAYERS     = 2       # number of stacked LSTM layers
DROPOUT_RATE    = 0.3     # dropout between LSTM layers
LEARNING_RATE   = 1e-3
BATCH_SIZE      = 32
EPOCHS          = 50      # early stopping will cut this short if val loss plateaus

# ─────────────────────────────────────────────
# DISPLAY / COMPOSITOR SETTINGS
# ─────────────────────────────────────────────

# Width of each panel in the tiled compositor window.
# Final window will be COMPOSITOR_PANEL_WIDTH × 2 wide.
COMPOSITOR_PANEL_WIDTH  = 640
COMPOSITOR_PANEL_HEIGHT = 480

# Number of recent commands shown in the command log overlay.
COMMAND_LOG_LENGTH = 3

# Skeleton drawing colour (BGR)
SKELETON_COLOR  = (0, 255, 0)    # green
TEXT_COLOR      = (255, 255, 255) # white
ALERT_COLOR     = (0, 0, 255)    # red — used when confidence is below threshold