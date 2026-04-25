from urllib.request import urlretrieve

from src.config import RAW_ARCHIVE_PATH, RAW_DIR

DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def download_raw_dataset(force: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_ARCHIVE_PATH.exists() and not force:
        print(f"Raw dataset already exists: {RAW_ARCHIVE_PATH}")
        return

    urlretrieve(DATA_URL, RAW_ARCHIVE_PATH)
    print(f"Downloaded raw dataset to {RAW_ARCHIVE_PATH}")


if __name__ == "__main__":
    download_raw_dataset()
