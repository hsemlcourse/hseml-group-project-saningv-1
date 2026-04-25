from pathlib import Path

RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "report"
IMAGES_DIR = REPORT_DIR / "images"

RAW_ARCHIVE_PATH = RAW_DIR / "sms_spam_collection.zip"
PROCESSED_DATA_PATH = PROCESSED_DIR / "sms_spam_processed.csv"
RESULTS_PATH = REPORT_DIR / "cp1_results.csv"
FINAL_MODEL_PATH = MODELS_DIR / "cp1_best_model.joblib"
