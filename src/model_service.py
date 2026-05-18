from dataclasses import dataclass
import json
from math import exp

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.config import FINAL_MODEL_PATH
from src.features import add_text_features
from src.train_baseline import load_or_build_processed, make_text_and_manual_pipeline, split_data


@dataclass(frozen=True)
class Prediction:
    label: str
    target: int
    spam_score: float


def train_deployment_model():
    df = load_or_build_processed()
    x_train, _, _, y_train, _, _ = split_data(df)
    model = make_text_and_manual_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2),
        LogisticRegression(max_iter=1000, random_state=42),
    )
    model.fit(x_train, y_train)
    FINAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, FINAL_MODEL_PATH)
    FINAL_MODEL_PATH.with_suffix(".json").write_text(
        json.dumps(
            {
                "model": "tfidf_logreg_with_manual_features",
                "purpose": "cp3 deployment",
                "selected_reason": "strong held-out F1 and native probability estimates for API/UI",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return model


def load_model():
    if FINAL_MODEL_PATH.exists():
        return joblib.load(FINAL_MODEL_PATH)
    return train_deployment_model()


def build_feature_frame(message: str) -> pd.DataFrame:
    base = pd.DataFrame({"label": ["unknown"], "message": [message], "target": [0]})
    features = add_text_features(base)
    return features.drop(columns=["label", "target"])


def score_spam(model, features: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(features)[0][1])

    if hasattr(model, "decision_function"):
        raw_score = float(model.decision_function(features)[0])
        return 1.0 / (1.0 + exp(-raw_score))

    return float(model.predict(features)[0])


def predict_message(message: str, model=None) -> Prediction:
    if not message or not message.strip():
        raise ValueError("Message must not be empty")

    active_model = model or load_model()
    features = build_feature_frame(message.strip())
    target = int(active_model.predict(features)[0])
    spam_score = score_spam(active_model, features)
    label = "spam" if target == 1 else "ham"
    return Prediction(label=label, target=target, spam_score=round(spam_score, 4))
