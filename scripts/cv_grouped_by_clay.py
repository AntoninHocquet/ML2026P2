"""Cross-validation groupée par argile (`GroupKFold` sur `parent_clay_id`).

Approche recommandée (sans fuite de données entre argiles) :
- Une même argile brute (`parent_clay_id`) n'apparaît jamais à la fois dans le
  jeu d'entraînement et dans le jeu de test.
- Tout le prétraitement (imputation par médiane + indicateur de manque,
  normalisation) est encapsulé dans un `Pipeline` entraîné uniquement sur le
  fold d'entraînement.
- Supporte également la validation Leave-One-Source-Out (`--loso`) sur
  `reference` ainsi que la Nested CV (`evaluate_nested_grouped_cv`) pour
  optimiser les hyperparamètres sans biais.

Utilisation en ligne de commande :
    python scripts/cv_grouped_by_clay.py
    python scripts/cv_grouped_by_clay.py --n-splits 5
    python scripts/cv_grouped_by_clay.py --dataset calcination --loso

Utilisation comme module dans vos propres scripts :
    from scripts.cv_grouped_by_clay import load_dataset, evaluate_grouped_cv
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
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut
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
    group_col: str = "parent_clay_id",
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Charge un dataset modélisation et sépare X, y et les groupes de CV."""
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in META_COLS + [TARGET_COL]]
    X = df[feature_cols].copy()
    y = df[TARGET_COL].to_numpy(dtype=float)
    groups = df[group_col].astype(str).to_numpy()
    return df, X, y, groups


def evaluate_grouped_cv(
    estimator: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    use_loso: bool = False,
) -> dict[str, Any]:
    """Évalue un estimateur par validation croisée groupée (0 fuite de groupe).

    Paramètres
    ----------
    estimator : estimateur ou Pipeline scikit-learn
    X : DataFrame des variables explicatives (sans META_COLS ni TARGET_COL)
    y : tableau 1D de la cible (r3_heat_7d_j_per_g)
    groups : tableau 1D des groupes (par défaut `parent_clay_id`)
    n_splits : nombre de plis pour `GroupKFold` (ignoré si `use_loso=True`)
    use_loso : si True, utilise `LeaveOneGroupOut` (utile sur `reference`)
    """
    splitter = LeaveOneGroupOut() if use_loso else GroupKFold(n_splits=n_splits)
    oof_pred = np.empty(len(y), dtype=float)
    fold_rmses: list[float] = []
    fold_sizes: list[int] = []

    for train_idx, test_idx in splitter.split(X, y, groups):
        # Garde-fou anti-leakage : aucun groupe ne doit être à la fois en train et en test
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        assert train_groups.isdisjoint(test_groups), (
            f"Fuite détectée entre train et test : {train_groups & test_groups}"
        )

        model = clone(estimator)
        model.fit(X.iloc[train_idx], y[train_idx])
        preds = model.predict(X.iloc[test_idx])

        oof_pred[test_idx] = preds
        fold_rmses.append(_rmse(y[test_idx], preds))
        fold_sizes.append(len(test_idx))

    return {
        "rmse_oof": _rmse(y, oof_pred),
        "mae_oof": float(mean_absolute_error(y, oof_pred)),
        "r2_oof": float(r2_score(y, oof_pred)),
        "rmse_fold_mean": float(np.mean(fold_rmses)),
        "rmse_fold_std": float(np.std(fold_rmses)),
        "fold_rmses": fold_rmses,
        "fold_sizes": fold_sizes,
        "leaked_test_rows": 0,
        "leaked_test_pct": 0.0,
        "oof_pred": oof_pred,
    }


def evaluate_nested_grouped_cv(
    estimator: Any,
    param_grid: dict[str, list[Any]] | list[dict[str, list[Any]]],
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    outer_splits: int = 5,
    inner_splits: int = 3,
    scoring: str = "neg_root_mean_squared_error",
) -> dict[str, Any]:
    """Validation croisée imbriquée (Nested CV) groupée pour le tuning d'hyperparamètres.

    La boucle externe estime honnêtement la généralisation ; la boucle interne
    (`GridSearchCV` + `GroupKFold`) sélectionne les hyperparamètres sur chaque
    fold d'entraînement uniquement.
    """
    outer_cv = GroupKFold(n_splits=outer_splits)
    oof_pred = np.empty(len(y), dtype=float)
    fold_rmses: list[float] = []
    best_params_per_fold: list[dict[str, Any]] = []

    for train_idx, test_idx in outer_cv.split(X, y, groups):
        assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))

        inner_cv = GroupKFold(n_splits=inner_splits)
        search = GridSearchCV(
            estimator=clone(estimator),
            param_grid=param_grid,
            cv=inner_cv,
            scoring=scoring,
            n_jobs=-1,
        )
        search.fit(X.iloc[train_idx], y[train_idx], groups=groups[train_idx])

        preds = search.best_estimator_.predict(X.iloc[test_idx])
        oof_pred[test_idx] = preds
        fold_rmses.append(_rmse(y[test_idx], preds))
        best_params_per_fold.append(search.best_params_)

    return {
        "rmse_oof": _rmse(y, oof_pred),
        "mae_oof": float(mean_absolute_error(y, oof_pred)),
        "r2_oof": float(r2_score(y, oof_pred)),
        "rmse_fold_mean": float(np.mean(fold_rmses)),
        "rmse_fold_std": float(np.std(fold_rmses)),
        "fold_rmses": fold_rmses,
        "best_params_per_fold": best_params_per_fold,
        "oof_pred": oof_pred,
    }


def get_default_models() -> dict[str, Any]:
    """Suite de modèles de référence (identique au script sans groupement)."""
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
    group_col: str = "parent_clay_id",
    use_loso: bool = False,
) -> pd.DataFrame:
    """Exécute la CV groupée sur les modèles de référence et affiche le bilan."""
    df, X, y, groups = load_dataset(dataset_path, group_col=group_col)
    scheme = (
        f"LeaveOneGroupOut sur `{group_col}` ({len(np.unique(groups))} groupes)"
        if use_loso
        else f"GroupKFold(n_splits={n_splits}) sur `{group_col}` ({len(np.unique(groups))} groupes)"
    )

    print("=" * 84)
    print(f"[CV GROUPÉE] Dataset: {dataset_name} ({len(df)} lignes, {X.shape[1]} features)")
    print(f"Stratégie : {scheme} | Fuite de groupe : 0 % (garanti par assert)")
    print("=" * 84)

    marsh_mask = df["parent_clay_id"].astype(str).str.startswith("Marsh-Marsh26_").to_numpy()
    has_marsh = bool(marsh_mask.any())

    rows = []
    for name, estimator in get_default_models().items():
        res = evaluate_grouped_cv(
            estimator, X, y, groups, n_splits=n_splits, use_loso=use_loso
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
        description="Cross-validation groupée par argile (ou par source avec --loso)."
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
        help="Nombre de folds pour GroupKFold (défaut: 5).",
    )
    parser.add_argument(
        "--group-col",
        choices=["parent_clay_id", "reference"],
        default="parent_clay_id",
        help="Colonne de groupement (défaut: parent_clay_id).",
    )
    parser.add_argument(
        "--loso",
        action="store_true",
        help="Active Leave-One-Source-Out (LeaveOneGroupOut sur `reference`).",
    )
    args = parser.parse_args()

    group_col = "reference" if args.loso else args.group_col
    selected = list(DATASETS.items()) if args.dataset == "all" else [(args.dataset, DATASETS[args.dataset])]

    for name, path in selected:
        run_benchmark(
            dataset_name=name,
            dataset_path=path,
            n_splits=args.n_splits,
            group_col=group_col,
            use_loso=args.loso,
        )


if __name__ == "__main__":
    main()
