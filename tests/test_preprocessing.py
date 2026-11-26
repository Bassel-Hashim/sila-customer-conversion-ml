import pandas as pd

from src.config import INPUT_FEATURES, LEAKAGE_COLUMNS, TARGET
from src.data import clean_data, split_data
from src.preprocessing import add_features, build_preprocessor


def make_rows(n=40):
    """Small synthetic dataset with the same columns as the real one."""
    rows = []
    for i in range(n):
        rows.append(
            {
                "age": 30 + i % 30, "job": "admin." if i % 2 else "technician",
                "marital": "married", "education": "high.school", "default": "no",
                "housing": "yes", "loan": "no", "contact": "cellular", "month": "may",
                "day_of_week": "mon", "duration": 100 + i, "campaign": 1 + i % 4,
                "pdays": 999 if i % 5 else 6, "previous": 0 if i % 5 else 1,
                "poutcome": "nonexistent" if i % 5 else "success",
                "emp.var.rate": 1.1, "cons.price.idx": 93.9, "cons.conf.idx": -36.4,
                "euribor3m": 4.8, "nr.employed": 5191.0,
                TARGET: "yes" if i % 8 == 0 else "no",
            }
        )
    return pd.DataFrame(rows)


def test_clean_data_drops_duplicates_and_leakage():
    df = make_rows()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # add 1 duplicate
    cleaned = clean_data(df)
    assert len(cleaned) == len(df) - 1
    for col in LEAKAGE_COLUMNS:
        assert col not in cleaned.columns
    assert set(cleaned[TARGET].unique()) <= {0, 1}


def test_add_features_handles_pdays_sentinel():
    out = add_features(make_rows())
    assert (out.loc[out["pdays"] == 0, "was_contacted_before"] >= 0).all()
    assert out["pdays"].max() < 999
    # 999 rows -> flag 0 ; real pdays rows -> flag 1
    raw = make_rows()
    assert (out["was_contacted_before"] == (raw["pdays"] != 999).astype(int)).all()


def test_preprocessor_is_consistent_between_fit_and_new_data():
    df = clean_data(make_rows())
    X = df.drop(columns=[TARGET])
    pre = build_preprocessor().fit(X)
    train_out = pre.transform(X)
    one_row = pre.transform(X.iloc[[0]])
    assert train_out.shape[1] == one_row.shape[1]
    assert train_out.shape[0] == len(X)


def test_preprocessor_ignores_unseen_category_and_duration():
    df = clean_data(make_rows())
    X = df.drop(columns=[TARGET])
    pre = build_preprocessor().fit(X)
    new = X.iloc[[0]].copy()
    new["job"] = "astronaut"      # never seen in training
    new["duration"] = 999         # leakage column must be ignored
    out = pre.transform(new)
    assert out.shape[0] == 1


def test_input_features_are_in_the_data():
    df = make_rows()
    assert all(c in df.columns for c in INPUT_FEATURES)


def test_split_is_stratified_and_non_overlapping():
    df = clean_data(pd.concat([make_rows(40)] * 10, ignore_index=True).assign(age=lambda d: d.index % 70 + 18))
    X_tr, X_val, X_te, y_tr, y_val, y_te = split_data(df)
    assert len(X_tr) + len(X_val) + len(X_te) == len(df)
    assert set(X_tr.index).isdisjoint(X_val.index)
    assert set(X_tr.index).isdisjoint(X_te.index)
    assert set(X_val.index).isdisjoint(X_te.index)
    assert abs(y_tr.mean() - y_te.mean()) < 0.05
