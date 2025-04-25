import os
import torch


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "outputs", "logs")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "outputs", "models", "best_model.pt")


NUM_CLASSES = 6
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
FREEZE_BACKBONE = False
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


NUM_WORKERS = 2


TUNING_EPOCHS = 5
TUNING_TRIALS = 10
TUNING_BATCH_SIZE = 32


LR_SCHEDULER_PATIENCE = 2
LR_SCHEDULER_FACTOR = 0.5


WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.3


DATA_AUG_ROTATION = 15
DATA_AUG_COLOR_JITTER = 0.1
DATA_AUG_TRANSLATE = 0.1
DATA_AUG_SCALE = (0.8, 1.0)


GRAD_CLIP_VALUE = 1.0


SALIENCY_METHODS = ["saliency", "smoothgrad", "guided"]
SMOOTHGRAD_SAMPLES = 20
SMOOTHGRAD_STDEV = 0.2


INFERENCE_DIR = os.path.join(DATA_DIR, "inference_test")


os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(INFERENCE_DIR, exist_ok=True)
