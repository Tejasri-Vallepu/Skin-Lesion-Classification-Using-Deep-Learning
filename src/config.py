
"""Central configuration for the HAM10000 project."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"

METADATA_FILE = DATA_DIR / "HAM10000_metadata.csv"
IMAGE_DIRS = [
    DATA_DIR / "HAM10000_images_part_1",
    DATA_DIR / "HAM10000_images_part_2",
]

# Keep this class order consistent across training, evaluation, and inference.
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
RANDOM_SEED = 42

CUSTOM_MODEL_PATH = MODEL_DIR / "skin_lesion_custom_cnn.keras"
TRANSFER_MODEL_PATH = MODEL_DIR / "mobilenetv2_finetuned_best.keras"
