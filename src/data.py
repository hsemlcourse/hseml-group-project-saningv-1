import zipfile
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DATA_PATH, PROCESSED_DIR, RAW_ARCHIVE_PATH
from src.features import add_text_features


def load_raw_sms(path: Path = RAW_ARCHIVE_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Raw archive was not found at {path}. Download it from UCI/Kaggle and keep the zip at this path."
        )

    with zipfile.ZipFile(path) as archive:
        with archive.open("SMSSpamCollection") as file:
            return pd.read_csv(file, sep="\t", names=["label", "message"], encoding="utf-8")


def clean_sms_data(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean["label"] = clean["label"].astype(str).str.strip().str.lower()
    clean["message"] = clean["message"].astype(str).str.strip()

    clean = clean[clean["label"].isin(["ham", "spam"])]
    clean = clean[clean["message"].ne("")]
    clean = clean.drop_duplicates(subset=["label", "message"]).reset_index(drop=True)
    clean["target"] = clean["label"].map({"ham": 0, "spam": 1}).astype(int)

    return add_text_features(clean)


def build_processed_dataset(
    raw_path: Path = RAW_ARCHIVE_PATH,
    output_path: Path = PROCESSED_DATA_PATH,
) -> pd.DataFrame:
    processed = clean_sms_data(load_raw_sms(raw_path))
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_path, index=False)
    return processed


if __name__ == "__main__":
    data = build_processed_dataset()
    print(f"Saved {len(data)} rows to {PROCESSED_DATA_PATH}")
