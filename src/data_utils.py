
"""Utilities for locating HAM10000 images and preparing metadata."""
import pandas as pd
from src.config import METADATA_FILE, IMAGE_DIRS, CLASS_NAMES

def load_metadata():
    """Load metadata and attach the local image path for each image ID."""
    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_FILE}. "
            "Place HAM10000_metadata.csv inside the data folder."
        )

    df = pd.read_csv(METADATA_FILE)
    required = {"image_id", "dx", "lesion_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")

    image_lookup = {}
    for folder in IMAGE_DIRS:
        if folder.exists():
            for path in folder.iterdir():
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}:
                    image_lookup[path.stem] = str(path)

    df["image_path"] = df["image_id"].map(image_lookup)
    df = df.dropna(subset=["image_path"]).copy()
    df = df[df["dx"].isin(CLASS_NAMES)].copy()
    return df.reset_index(drop=True)
