import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model_service import load_model, predict_message  # noqa: E402

st.set_page_config(page_title="SMS Spam Detection", page_icon="✉️", layout="centered")

MODEL = load_model()

EXAMPLES = {
    "Обычное сообщение": "Hey, are we still meeting near the station at 7?",
    "Похоже на спам": "URGENT! You won a free prize. Call 09061701461 now to claim.",
}

st.title("SMS Spam Detection")
st.caption("Локальный интерфейс для проверки финальной модели проекта.")

example = st.selectbox("Пример", ["Свой текст", *EXAMPLES.keys()])
default_message = "" if example == "Свой текст" else EXAMPLES[example]
message = st.text_area("SMS-сообщение", value=default_message, height=140)

if st.button("Проверить", type="primary"):
    if not message.strip():
        st.warning("Введите текст сообщения.")
    else:
        prediction = predict_message(message, MODEL)
        if prediction.label == "spam":
            st.error(f"Spam. Score: {prediction.spam_score:.3f}")
        else:
            st.success(f"Ham. Score: {prediction.spam_score:.3f}")

        st.metric("Класс", prediction.label)
        st.metric("Spam score", f"{prediction.spam_score:.3f}")
