import matplotlib.pyplot as plt
import pandas as pd

from src.config import IMAGES_DIR, REPORT_DIR
from src.data import build_processed_dataset


def save_eda_artifacts() -> None:
    df = build_processed_dataset()
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    class_counts = df["label"].value_counts().sort_index()
    class_counts.plot(kind="bar", color=["#4C78A8", "#F58518"])
    plt.title("Class balance")
    plt.xlabel("Label")
    plt.ylabel("Messages")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "class_balance.png", dpi=160)
    plt.close()

    df.boxplot(column="message_length", by="label")
    plt.title("Message length by class")
    plt.suptitle("")
    plt.xlabel("Label")
    plt.ylabel("Characters")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "message_length_by_class.png", dpi=160)
    plt.close()

    df.groupby("label")[["digit_count", "cta_word_count", "has_phone", "has_currency"]].mean().plot(kind="bar")
    plt.title("Mean manual spam indicators")
    plt.xlabel("Label")
    plt.ylabel("Mean value")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "manual_feature_means.png", dpi=160)
    plt.close()

    summary = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicates_after_cleaning": int(df.duplicated(subset=["label", "message"]).sum()),
        "class_counts": class_counts.to_dict(),
    }
    pd.Series(summary).to_json(REPORT_DIR / "cp1_data_summary.json", force_ascii=False, indent=2)


if __name__ == "__main__":
    save_eda_artifacts()
