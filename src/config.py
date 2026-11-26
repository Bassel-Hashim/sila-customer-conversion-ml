"""Central place for paths, constants and column lists.

Every other module imports from here, so a change (e.g. a new seed or a new
column) is made in exactly one place.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "raw" / "bank-additional-full.csv"
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

# ---------------------------------------------------------- reproducibility
RANDOM_STATE = 42

# --------------------------------------------------------------- target
TARGET = "y"

# Columns that must NOT be used as features (data leakage).
# `duration` = length of the call. It is only known AFTER the call, so it
# cannot be used to decide whom to call.
LEAKAGE_COLUMNS = ["duration"]

# ---------------------------------------------------------------- split
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# -------------------------------------------------------------- features
# `pdays == 999` is a sentinel meaning "client was never contacted before".
PDAYS_NEVER_CONTACTED = 999

NUMERIC_FEATURES = [
    "age",
    "campaign",
    "pdays",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]

CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]

# Created inside the pipeline (see preprocessing.add_features).
ENGINEERED_BINARY_FEATURES = ["was_contacted_before"]

# What a caller (training data or the API) must provide for each customer.
INPUT_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
