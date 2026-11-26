"""Reusable preprocessing pipeline.

The same object is used for training and (later) inside the API, so
preprocessing is identical at training time and prediction time.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    ENGINEERED_BINARY_FEATURES,
    NUMERIC_FEATURES,
    PDAYS_NEVER_CONTACTED,
)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level feature engineering.

    `pdays` uses 999 to mean "never contacted before". Treating 999 as a real
    number would be wrong (it would look like 'contacted 999 days ago').
    So we:
      * create a clear 0/1 flag `was_contacted_before`
      * set pdays to 0 for never-contacted clients (the flag tells them apart)
    """
    df = df.copy()
    never_contacted = df["pdays"] == PDAYS_NEVER_CONTACTED
    df["was_contacted_before"] = (~never_contacted).astype(int)
    df["pdays"] = df["pdays"].where(~never_contacted, 0)
    return df


def build_preprocessor() -> Pipeline:
    """Feature engineering + scaling + one-hot encoding in one object.

    * numeric columns  -> StandardScaler (needed by Logistic Regression)
    * categorical cols -> OneHotEncoder. 'unknown' is kept as its own category
      (it can be informative). handle_unknown='ignore' stops the API from
      crashing if it ever sees a category it did not see in training.
    * the 0/1 flag     -> passed through unchanged
    * every other column (e.g. leftover `duration`) is dropped.
    """
    encoder = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("flag", "passthrough", ENGINEERED_BINARY_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("add_features", FunctionTransformer(add_features)),
            ("encode", encoder),
        ]
    )
