"""Cross-validation aléatoire SANS groupement par argile (`KFold` classique).

ATTENTION — Script de comparaison / contre-exemple pour le rapport :
- Découpe les lignes aléatoirement (`KFold(shuffle=True)`) SANS tenir compte
  de `parent_clay_id`.
- Une même argile brute (même composition chimique/minéralogique) peut donc
  apparaître à la fois dans le jeu d'entraînement et dans le jeu de test
  (fuite de données / data leakage).
- Les modèles et le prétraitement (`Pipeline`) sont STRICTEMENT identiques à
  `scripts/cv_grouped_by_clay.py` afin d'isoler uniquement l'effet du
  groupement par argile.
- Mesure explicitement combien de lignes de test appartiennent à une argile
  déjà vue pendant l'entraînement.

Utilisation en ligne de commande :
    python scripts/cv_random_kfold_no_group.py
    python scripts/cv_random_kfold_no_group.py --n-splits 5 --seed 42
    python scripts/cv_random_kfold_no_group.py --dataset mechanical

Utilisation comme module :
    from scripts.cv_random_kfold_no_group import load_dataset, evaluate_random_cv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "research" / "clay-activation-r3" / "processed"
DATASETS = {
    "calcination": PROCESSED_DIR / "calcination_dataset.csv",
    "mechanical": PROCESSED_DIR / "mechanical_dataset.csv",
}

META_COLS = ["sample_id", "parent_clay_id", "reference"]
TARGET_COL = "r3_heat_7d_j_per_g"


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcule la RMSE (compatible toutes versions de scikit-learn)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_dataset(
    path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Charge un dataset modélisation et sépare X, y et les identifiants d'argile.

    Note : `clay_ids` n'est PAS utilisé pour découper les folds dans ce script ;
    il sert uniquement à mesurer le taux de fuite d'argiles entre train et test.
    """
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in META_COLS + [TARGET_COL]]
    X = df[feature_cols].copy()
    y = df[TARGET_COL].to_numpy(dtype=float)
    clay_ids = df["parent_clay_id"].astype(str).to_numpy()
    return df, X, y, clay_ids


def evaluate_random_cv(
    estimator: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    clay_ids: np.ndarray | None = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """Évalue un estimateur par KFold aléatoire SANS groupement par argile.

    Paramètres
    ----------
    estimator : estimateur ou Pipeline scikit-learn
    X : DataFrame des variables explicatives (sans META_COLS ni TARGET_COL)
    y : tableau 1D de la cible (r3_heat_7d_j_per_g)
    clay_ids : optionnel, `parent_clay_id` pour quantifier la fuite train/test
    n_splits : nombre de plis pour `KFold`
    random_state : graine aléatoire pour la reproductibilité du mélange
    """
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_pred = np.empty(len(y), dtype=float)
    fold_rmses: list[float] = []
    fold_sizes: list[int] = []
    leaked_rows_total = 0
    leaked_clays_per_fold: list[int] = []

    for train_idx, test_idx in splitter.split(X, y):
        if clay_ids is not None:
            train_clays = set(clay_ids[train_idx])
            test_clays_arr = clay_ids[test_idx]
            leaked_mask = np.isin(test_clays_arr, list(train_clays))
            leaked_rows_total += int(leaked_mask.sum())
            leaked_clays_per_fold.append(len(train_clays & set(test_clays_arr)))

        model = clone(estimator)
        model.fit(X.iloc[train_idx], y[train_idx])
        preds = model.predict(X.iloc[test_idx])

        oof_pred[test_idx] = preds
        fold_rmses.append(_rmse(y[test_idx], preds))
        fold_sizes.append(len(test_idx))

    leaked_pct = 100.0 * leaked_rows_total / len(y) if clay_ids is not None else np.nan

    return {
        "rmse_oof": _rmse(y, oof_pred),
        "mae_oof": float(mean_absolute_error(y, oof_pred)),
        "r2_oof": float(r2_score(y, oof_pred)),
        "rmse_fold_mean": float(np.mean(fold_rmses)),
        "rmse_fold_std": float(np.std(fold_rmses)),
        "fold_rmses": fold_rmses,
        "fold_sizes": fold_sizes,
        "leaked_test_rows": leaked_rows_total,
        "leaked_test_pct": float(leaked_pct),
        "leaked_clays_per_fold": leaked_clays_per_fold,
        "oof_pred": oof_pred,
    }


def get_default_models() -> dict[str, Any]:
    """Suite de modèles de référence (STRICTEMENT identique à `cv_grouped_by_clay.py`)."""
    return {
        "Dummy (mean)": DummyRegressor(strategy="mean"),
        "Ridge": Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
        ]),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=200,
            max_depth=4,
            min_samples_leaf=8,
            learning_rate=0.05,
            random_state=42,
        ),
    }


def run_benchmark(
    dataset_name: str,
    dataset_path: Path,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Exécute le KFold aléatoire (sans groupement) et quantifie la fuite d'argiles."""
    df, X, y, clay_ids = load_dataset(dataset_path)
    n_clays = len(np.unique(clay_ids))

    # Calcul préalable du taux de fuite sur ce découpage aléatoire
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    leaked_rows = 0
    for train_idx, test_idx in splitter.split(X, y):
        train_clays = set(clay_ids[train_idx])
        leaked_rows += int(np.isin(clay_ids[test_idx], list(train_clays)).sum())
    leaked_pct = 100.0 * leaked_rows / len(df)

    print("=" * 84)
    print(f"[CV ALÉATOIRE - SANS GROUPEMENT] Dataset: {dataset_name} ({len(df)} lignes, {n_clays} argiles)")
    print(f"Stratégie : KFold(n_splits={n_splits}, shuffle=True, random_state={random_state})")
    print(
        f"Fuite d'argiles mesurée : {leaked_rows}/{len(df)} lignes de test ({leaked_pct:.1f} %) "
        "appartiennent à une argile déjà présente dans le train !"
    )
    print("=" * 84)

    marsh_mask = df["parent_clay_id"].astype(str).str.startswith("Marsh-Marsh26_").to_numpy()
    has_marsh = bool(marsh_mask.any())

    rows = []
    for name, estimator in get_default_models().items():
        res = evaluate_random_cv(
            estimator,
            X,
            y,
            clay_ids=clay_ids,
            n_splits=n_splits,
            random_state=random_state,
        )
        row = {
            "model": name,
            "rmse_oof": round(res["rmse_oof"], 1),
            "mae_oof": round(res["mae_oof"], 1),
            "r2_oof": round(res["r2_oof"], 3),
            "rmse_fold_mean": round(res["rmse_fold_mean"], 1),
            "rmse_fold_std": round(res["rmse_fold_std"], 1),
        }
        if has_marsh:
            row["rmse_hors_marsh26"] = round(
                _rmse(y[~marsh_mask], res["oof_pred"][~marsh_mask]), 1
            )
        rows.append(row)

    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    print()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-validation aléatoire SANS groupement par argile (comparaison rapport)."
    )
    parser.add_argument(
        "--dataset",
        choices=["all", "calcination", "mechanical"],
        default="all",
        help="Dataset à évaluer (défaut: all).",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Nombre de folds pour KFold (défaut: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Graine aléatoire pour le mélange KFold (défaut: 42).",
    )
    args = parser.parse_args()

    selected = list(DATASETS.items()) if args.dataset == "all" else [(args.dataset, DATASETS[args.dataset])]

    for name, path in selected:
        run_benchmark(
            dataset_name=name,
            dataset_path=path,
            n_splits=args.n_splits,
            random_state=args.seed,
        )


if __name__ == "__main__":
    main()
