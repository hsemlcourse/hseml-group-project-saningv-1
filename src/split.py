import pandas as pd


def stratified_train_val_test_split(
    df: pd.DataFrame,
    target_column: str,
    random_state: int,
    train_size: float = 0.70,
    val_size: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test_size = 1.0 - train_size - val_size
    if train_size <= 0 or val_size <= 0 or test_size <= 0:
        raise ValueError("train_size, val_size and derived test_size must be positive")

    train_parts = []
    val_parts = []
    test_parts = []

    for _, group in df.groupby(target_column, group_keys=False):
        shuffled = group.sample(frac=1.0, random_state=random_state)
        train_end = int(round(len(shuffled) * train_size))
        val_end = train_end + int(round(len(shuffled) * val_size))

        train_parts.append(shuffled.iloc[:train_end])
        val_parts.append(shuffled.iloc[train_end:val_end])
        test_parts.append(shuffled.iloc[val_end:])

    train = pd.concat(train_parts).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    val = pd.concat(val_parts).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    test = pd.concat(test_parts).sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    return train, val, test
