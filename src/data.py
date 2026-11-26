"""Loading, cleaning and splitting the Bank Marketing data."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_PATH,
    LEAKAGE_COLUMNS,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
    VAL_SIZE,
)
from src.logger import get_logger

logger = get_logger(__name__)


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV (note: the file is separated by ';', not ',')."""
    df = pd.read_csv(path, sep=";")
    logger.info("Loaded %s rows x %s columns from %s", *df.shape, path.name)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Dataset-level cleaning. Used for TRAINING data only.

    1. Drop exact duplicate rows.
    2. Drop leakage columns (`duration`).
    3. Turn the target 'yes'/'no' into 1/0.

    Row-level feature logic (e.g. the pdays flag) lives in
    preprocessing.add_features so the API can reuse it.
    """
    df = df.copy()

    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info("Dropped %s duplicate rows", n_before - len(df))

    df = df.drop(columns=LEAKAGE_COLUMNS, errors="ignore")
    logger.info("Dropped leakage columns: %s", LEAKAGE_COLUMNS)

    df[TARGET] = (df[TARGET] == "yes").astype(int)
    return df


def split_data(df: pd.DataFrame):
    """Stratified train / validation / test split (70 / 15 / 15).

    Stratified = every part keeps the same ~11% conversion rate.
    The test set is only used once, at the very end.
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # 1) carve out the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    # 2) split the rest into train and validation
    val_fraction = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_fraction, stratify=y_temp, random_state=RANDOM_STATE
    )

    logger.info(
        "Split sizes -> train: %s | val: %s | test: %s",
        len(X_train), len(X_val), len(X_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_splits():
    """Convenience: load -> clean -> split in one call."""
    return split_data(clean_data(load_data()))
