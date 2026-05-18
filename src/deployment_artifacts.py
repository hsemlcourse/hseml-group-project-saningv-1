import json

import matplotlib.pyplot as plt

from src.config import IMAGES_DIR
from src.model_service import predict_message


def save_text_panel(path, title: str, lines: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.axis("off")
    ax.text(0.03, 0.90, title, fontsize=18, weight="bold", va="top")
    ax.text(
        0.03,
        0.76,
        "\n".join(lines),
        fontsize=12,
        family="monospace",
        va="top",
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "#f7f7f7", "edgecolor": "#d0d0d0"},
    )
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)


def save_deployment_artifacts() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    spam_message = "URGENT! You won a free prize. Call now to claim cash."
    ham_message = "Hey, are we still meeting near the station at 7?"
    spam_prediction = predict_message(spam_message)
    ham_prediction = predict_message(ham_message)

    save_text_panel(
        IMAGES_DIR / "api_predict_example.png",
        "FastAPI /predict",
        [
            "POST /predict",
            json.dumps({"message": spam_message}, ensure_ascii=False),
            "",
            "Response",
            json.dumps(spam_prediction.__dict__, ensure_ascii=False, indent=2),
        ],
    )
    save_text_panel(
        IMAGES_DIR / "streamlit_interface_example.png",
        "Streamlit UI",
        [
            "SMS Spam Detection",
            "",
            f"Input: {ham_message}",
            f"Prediction: {ham_prediction.label}",
            f"Spam score: {ham_prediction.spam_score:.4f}",
            "",
            f"Input: {spam_message}",
            f"Prediction: {spam_prediction.label}",
            f"Spam score: {spam_prediction.spam_score:.4f}",
        ],
    )


if __name__ == "__main__":
    save_deployment_artifacts()
