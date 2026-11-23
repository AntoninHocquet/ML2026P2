"""What this module does:
  - load a processed dataset (calcination or mechanical);
  - add engineered process features (log terms, centred temperature and its square, TEA flag);
  - add missingness indicators for the chosen columns, one per distinct missingness pattern.

Example:
    from data import load_dataset, feature_list, add_missing_indicators, make_imputer, TARGET
    df = load_dataset("research/clay-activation-r3/processed", "mechanical")
    df, cols = add_missing_indicators(df, feature_list("mechanical", "process+mineralogy"))
    pipe = Pipeline([("impute", make_imputer()), ("ols", LinearRegression())])
    pipe.fit(df[cols], df[TARGET])
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

TARGET = "r3_heat_7d_j_per_g"
CLAY = "parent_clay_id"
SOURCE = "reference"

KINDS = ("calcination", "mechanical")
FEATURE_SETS = ("process", "process+mineralogy", "all")

MINERALOGY = ["kaolinite_pct", "illite_mica_pct", "quartz_pct", "calcite_pct"]
OXIDES = ["sio2_pct", "al2o3_pct", "fe2o3_pct", "cao_pct", "mgo_pct", "k2o_pct", "loi_pct"]

PROCESS = {
    "calcination": ["temp_c_centered_100", "temp_c_centered_100_sq", "log_dwell_min"],
    "mechanical": ["log_milling_time_min", "log_mill_rpm", "log_bpr", "grinding_aid_used"],
}

MISSING_PREFIX = "missing_"


def add_process_features(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Engineered process terms (returns a copy)."""
    df = df.copy()
    if kind == "calcination":
        t = (df["calcination_temp_c"] - 800.0) / 100.0
        df["temp_c_centered_100"] = t
        df["temp_c_centered_100_sq"] = t**2
        df["log_dwell_min"] = np.log(df["dwell_min"])
    elif kind == "mechanical":
        df["log_milling_time_min"] = np.log(df["milling_time_min"])
        df["log_mill_rpm"] = np.log(df["mill_rpm"])
        df["log_bpr"] = np.log(df["ball_to_powder_ratio"])
        df["grinding_aid_used"] = (df["grinding_aid_tea_wt_pct"] > 0).astype(float)
    else:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    return df


def feature_list(kind: str, feature_set: str) -> list[str]:
    """Feature columns of a feature set (without missingness indicators)."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {FEATURE_SETS}, got {feature_set!r}")
    cols = list(PROCESS[kind])
    if feature_set in ("process+mineralogy", "all"):
        cols += MINERALOGY
    if feature_set == "all":
        cols += OXIDES
    return cols


def add_missing_indicators(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Add one 0/1 column per distinct missingness pattern among `cols`.

    Returns a copy of `df` with the indicators, and `cols` + the indicator names.

    Columns that are always missing together (e.g. all oxides of a clay without XRF) share one
    indicator; separate ones would be exactly collinear.
    """
    df = df.copy()
    patterns: dict[tuple, list[str]] = {}
    for c in cols:
        m = df[c].isna()
        if m.any():
            patterns.setdefault(tuple(m), []).append(c)
    names = []
    for group in patterns.values():
        label = "+".join(group) if len(group) <= 2 else f"{group[0]}..{group[-1]}({len(group)})"
        df[MISSING_PREFIX + label] = df[group[0]].isna().astype(float)
        names.append(MISSING_PREFIX + label)
    return df, cols + names


def load_dataset(processed_dir, kind: str) -> pd.DataFrame:
    """Read `<kind>_dataset.csv` and add the process features. Missing values stay NaN."""
    df = pd.read_csv(Path(processed_dir) / f"{kind}_dataset.csv")
    return add_process_features(df, kind)


def make_imputer() -> SimpleImputer:
    """Unfitted median imputer. Put it in a Pipeline so it is fitted per CV fold."""
    return SimpleImputer(strategy="median")