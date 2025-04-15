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
