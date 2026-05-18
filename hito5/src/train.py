"""Pipeline de entrenamiento del sistema CARDIS.

Carga el dataset crudo, ejecuta la ingenieria de caracteristicas del
Hito 2, entrena LightGBM con los hiperparametros optimos del Hito 3,
calibra las probabilidades con Isotonic Regression y persiste todos los
artefactos. Si MLflow esta configurado, registra el experimento y los
artefactos resultantes (parametros, metricas y modelo).

Ejemplo de uso (linea de comandos)::

    python -m src.train \
        --input data/raw/cardis_train.csv \
        --model-path models/cardis_lightgbm.joblib

El contenedor ``trainer`` lo invoca como punto de entrada por defecto.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import MLflowConfig, ModelConfig, Paths, StorageConfig, setup_logging
from .data_ingestion import load_csv, load_from_minio
from .features import FeatureBuilder, IsotonicCalibrator, split_xy, write_feature_metadata
from .model_store import ModelStore

logger = setup_logging()


def cross_validated_metrics(
    X: pd.DataFrame, y: pd.Series, cfg: ModelConfig
) -> Dict[str, float]:
    """Realiza una validacion cruzada estratificada para reportar metricas."""
    skf = StratifiedKFold(
        n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state
    )
    metrics = {"f1": [], "recall": [], "precision": [], "auc": [], "brier": []}
    for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
        model = LGBMClassifier(**cfg.lgbm_params)
        model.fit(X.iloc[tr], y.iloc[tr])
        proba = model.predict_proba(X.iloc[va])[:, 1]
        preds = (proba >= cfg.threshold).astype(int)
        metrics["f1"].append(f1_score(y.iloc[va], preds))
        metrics["recall"].append(recall_score(y.iloc[va], preds))
        metrics["precision"].append(precision_score(y.iloc[va], preds, zero_division=0))
        metrics["auc"].append(roc_auc_score(y.iloc[va], proba))
        metrics["brier"].append(brier_score_loss(y.iloc[va], proba))
        logger.info(
            "Fold %d/%d -> F1=%.4f Recall=%.4f Precision=%.4f AUC=%.4f Brier=%.4f",
            fold,
            cfg.n_splits,
            metrics["f1"][-1],
            metrics["recall"][-1],
            metrics["precision"][-1],
            metrics["auc"][-1],
            metrics["brier"][-1],
        )
    return {f"{k}_mean": float(np.mean(v)) for k, v in metrics.items()} | {
        f"{k}_std": float(np.std(v)) for k, v in metrics.items()
    }


def fit_final_model(
    X: pd.DataFrame, y: pd.Series, cfg: ModelConfig
) -> tuple[LGBMClassifier, IsotonicCalibrator]:
    """Entrena LightGBM final + calibrador isotonico."""
    X_tr, X_va, y_tr, y_va = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )
    base = LGBMClassifier(**cfg.lgbm_params)
    base.fit(X_tr, y_tr)
    logger.info("Modelo base entrenado sobre %d filas", len(X_tr))

    # IsotonicRegression directa sobre el conjunto de validacion (equivalente a
    # CalibratedClassifierCV(cv='prefit'), API eliminada en sklearn >= 1.6).
    proba_va_raw = base.predict_proba(X_va)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(proba_va_raw, y_va)
    calibrator = IsotonicCalibrator(base, iso)
    logger.info("Calibrador isotonico ajustado sobre %d filas", len(X_va))

    proba = calibrator.predict_proba(X_va)[:, 1]
    preds = (proba >= cfg.threshold).astype(int)
    holdout = {
        "f1": f1_score(y_va, preds),
        "recall": recall_score(y_va, preds),
        "precision": precision_score(y_va, preds, zero_division=0),
        "auc": roc_auc_score(y_va, proba),
        "brier": brier_score_loss(y_va, proba),
        "average_precision": average_precision_score(y_va, proba),
    }
    logger.info("Hold-out metrics: %s", holdout)
    return base, calibrator, holdout  # type: ignore[return-value]


def maybe_log_mlflow(
    cfg: ModelConfig,
    metrics: Dict[str, float],
    holdout: Dict[str, float],
    artifacts: Dict[str, Path],
) -> None:
    mlf = MLflowConfig()
    if not mlf.enabled:
        logger.info("MLflow desactivado (falta MLFLOW_TRACKING_URI)")
        return
    try:
        import mlflow  # type: ignore
    except ImportError:
        logger.warning("MLflow no instalado, se omite el logging")
        return
    mlflow.set_tracking_uri(mlf.tracking_uri)
    mlflow.set_experiment(mlf.experiment_name)
    with mlflow.start_run(run_name=f"cardis-{int(time.time())}"):
        mlflow.log_params(cfg.lgbm_params)
        mlflow.log_param("threshold", cfg.threshold)
        mlflow.log_param("n_splits", cfg.n_splits)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        for k, v in holdout.items():
            mlflow.log_metric(f"holdout_{k}", v)
        for name, path in artifacts.items():
            mlflow.log_artifact(str(path), artifact_path=name)


def run(args: argparse.Namespace) -> None:
    cfg = ModelConfig()
    paths = Paths()
    paths.ensure()

    if os.getenv("LOAD_FROM_MINIO", "false").lower() == "true":
        df = load_from_minio(
            bucket=os.getenv("MINIO_BUCKET", "cardis"),
            key=os.getenv("MINIO_KEY", "cardis_train.csv")
        )
    else:
        df = load_csv(args.input)
    builder = FeatureBuilder(target_col=cfg.target_col)
    df_features = builder.fit_transform(df)
    X, y = df_features, df[cfg.target_col].astype(int)
    logger.info(
        "Dataset preparado: %d filas, %d features", X.shape[0], X.shape[1]
    )

    metrics = cross_validated_metrics(X, y, cfg)
    logger.info("Metricas CV: %s", metrics)

    base_model, calibrator, holdout = fit_final_model(X, y, cfg)

    # Persistir artefactos
    model_path = Path(args.model_path)
    calibrator_path = Path(args.calibrator_path)
    builder_path = paths.model_dir / "cardis_featurebuilder.joblib"
    metadata_path = Path(args.metadata_path)

    joblib.dump(base_model, model_path)
    joblib.dump(calibrator, calibrator_path)
    builder.save(builder_path)

    metadata: Dict[str, Any] = {
        "model_name": "cardis-lightgbm",
        "model_version": "1.0.0",
        "threshold": cfg.threshold,
        "metrics_cv": metrics,
        "metrics_holdout": holdout,
        "feature_names": builder.feature_names_,
        "lgbm_params": cfg.lgbm_params,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)
    write_feature_metadata(builder, paths.model_dir / "feature_schema.json")
    logger.info("Artefactos guardados en %s", paths.model_dir)

    # Subir a MinIO si esta activo
    storage_cfg = StorageConfig()
    if storage_cfg.backend == "minio":
        store = ModelStore(storage_cfg)
        for path in (model_path, calibrator_path, builder_path, metadata_path):
            store.put(path, remote_name=path.name)

    maybe_log_mlflow(
        cfg,
        metrics,
        holdout,
        artifacts={
            "model": model_path,
            "calibrator": calibrator_path,
            "feature_builder": builder_path,
            "metadata": metadata_path,
        },
    )
    logger.info("Entrenamiento completado correctamente")


def _build_parser() -> argparse.ArgumentParser:
    paths = Paths()
    parser = argparse.ArgumentParser(description="Entrenamiento de CARDIS")
    parser.add_argument("--input", type=Path, default=paths.raw_train)
    parser.add_argument("--model-path", type=Path, default=paths.model_path)
    parser.add_argument(
        "--calibrator-path", type=Path, default=paths.calibrator_path
    )
    parser.add_argument("--metadata-path", type=Path, default=paths.metadata_path)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
