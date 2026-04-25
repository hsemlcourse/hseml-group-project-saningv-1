import pandas as pd

from src.data import clean_sms_data
from src.features import add_text_features
from src.split import stratified_train_val_test_split


def test_feature_engineering_adds_expected_columns():
    df = pd.DataFrame(
        {
            "label": ["spam"],
            "message": ["URGENT! Call 08001234567 now to claim £1000"],
        }
    )

    result = add_text_features(df)

    assert result.loc[0, "has_phone"] == 1
    assert result.loc[0, "has_currency"] == 1
    assert result.loc[0, "cta_word_count"] >= 2
    assert result.loc[0, "message_length"] > 0


def test_clean_sms_data_removes_duplicate_messages_and_encodes_target():
    raw = pd.DataFrame(
        {
            "label": ["ham", "ham", "spam"],
            "message": ["hello", "hello", "free prize"],
        }
    )

    result = clean_sms_data(raw)

    assert len(result) == 2
    assert set(result["target"]) == {0, 1}
    assert result["message"].isna().sum() == 0


def test_split_data_keeps_both_classes_in_each_split():
    rows = []
    for index in range(40):
        rows.append({"label": "ham", "message": f"regular message {index}", "target": 0})
    for index in range(20):
        rows.append({"label": "spam", "message": f"free prize call {index}", "target": 1})

    df = add_text_features(pd.DataFrame(rows))
    train, val, test = stratified_train_val_test_split(df, "target", random_state=42)

    assert set(train["target"]) == {0, 1}
    assert set(val["target"]) == {0, 1}
    assert set(test["target"]) == {0, 1}
