import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.config import FINAL_MODEL_PATH, MODELS_DIR, PROCESSED_DATA_PATH, RANDOM_STATE, RESULTS_PATH
from src.data import build_processed_dataset
from src.split import stratified_train_val_test_split

TEXT_FEATURES = [
    "message_length",
    "word_count",
    "digit_count",
    "uppercase_count",
    "uppercase_ratio",
    "exclamation_count",
    "question_count",
    "special_char_count",
    "has_url",
    "has_phone",
    "has_currency",
    "cta_word_count",
]


def load_or_build_processed() -> pd.DataFrame:
    if PROCESSED_DATA_PATH.exists():
        return pd.read_csv(PROCESSED_DATA_PATH)
    return build_processed_dataset()


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    train, val, test = stratified_train_val_test_split(df, "target", RANDOM_STATE)

    x_train = train[["message", *TEXT_FEATURES]]
    x_val = val[["message", *TEXT_FEATURES]]
    x_test = test[["message", *TEXT_FEATURES]]

    return x_train, x_val, x_test, train["target"], val["target"], test["target"]


def make_text_only_pipeline(vectorizer, model) -> Pipeline:
    return Pipeline(
        steps=[
            ("text", vectorizer),
            ("model", model),
        ]
    )


def make_text_and_manual_pipeline(vectorizer, model) -> Pipeline:
    features = ColumnTransformer(
        transformers=[
            ("text", vectorizer, "message"),
            ("manual", StandardScaler(), TEXT_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("features", features),
            ("model", model),
        ]
    )


def evaluate(model: Pipeline, x: pd.DataFrame | pd.Series, y: pd.Series) -> dict[str, float]:
    prediction = model.predict(x)
    return {
        "accuracy": accuracy_score(y, prediction),
        "precision_spam": precision_score(y, prediction, zero_division=0),
        "recall_spam": recall_score(y, prediction, zero_division=0),
        "f1_spam": f1_score(y, prediction, zero_division=0),
    }


def run_experiments() -> pd.DataFrame:
    df = load_or_build_processed()
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(df)

    experiments = {
        "count_nb_text_only": make_text_only_pipeline(CountVectorizer(), MultinomialNB()),
        "tfidf_logreg_text_only": make_text_only_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2),
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        ),
        "tfidf_linearsvc_text_only": make_text_only_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2),
            LinearSVC(random_state=RANDOM_STATE),
        ),
        "tfidf_logreg_with_manual_features": make_text_and_manual_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2),
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        ),
    }

    rows = []
    fitted_models = {}
    for name, model in experiments.items():
        if name.endswith("text_only"):
            model.fit(x_train["message"], y_train)
            val_metrics = evaluate(model, x_val["message"], y_val)
            test_metrics = evaluate(model, x_test["message"], y_test)
        else:
            model.fit(x_train, y_train)
            val_metrics = evaluate(model, x_val, y_val)
            test_metrics = evaluate(model, x_test, y_test)

        rows.append({"model": name, "split": "val", **val_metrics})
        rows.append({"model": name, "split": "test", **test_metrics})
        fitted_models[name] = model

    results = pd.DataFrame(rows).sort_values(["split", "f1_spam"], ascending=[True, False])
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)

    best_name = (
        results[results["split"].eq("val")]
        .sort_values("f1_spam", ascending=False)
        .iloc[0]["model"]
    )
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted_models[best_name], FINAL_MODEL_PATH)

    metadata_path = FINAL_MODEL_PATH.with_suffix(".json")
    metadata_path.write_text(
        json.dumps({"selected_by": "best validation f1_spam", "model": best_name}, indent=2),
        encoding="utf-8",
    )
    return results


if __name__ == "__main__":
    metrics = run_experiments()
    print(metrics.to_string(index=False))
