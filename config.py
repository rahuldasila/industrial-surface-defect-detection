from pathlib import Path
import torch

# Project root (folder containing config.py)
PROJECT_ROOT = Path(__file__).resolve().parent

# Dataset
DATA_DIR = PROJECT_ROOT / "data" / "KolektorSDD2"

TRAIN_PATH = DATA_DIR / "train"
TEST_PATH = DATA_DIR / "test"

# Model
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "best_model.pth"

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training parameters
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 20