"""Configuración global del sistema CARDIS.

Centraliza la lectura de variables de entorno y argumentos de línea de
comandos, de modo que ningún módulo escriba rutas o credenciales en
duro. 
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Directorio compartido para métricas en modo multiprocess de prometheus_client.
# Debe existir antes de que se importe prometheus_client (la librería lo lee
# al importarse, no al usarse). setdefault para que un valor externo (Docker ENV,
# K8s, etc.) tenga prioridad. Hito 5 — Bloque 1.
PROMETHEUS_MULTIPROC_DIR = os.environ.setdefault(
    "PROMETHEUS_MULTIPROC_DIR", "/tmp/prom"
)
Path(PROMETHEUS_MULTIPROC_DIR).mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def setup_logging(level: str | None = None) -> logging.Logger:
    """Configura el logger raíz del sistema con un formato uniforme."""
    level = (level or os.getenv("CARDIS_LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("cardis")


# -----------------------------------------------------------------------------
# Rutas de datos / modelos
# -----------------------------------------------------------------------------
@dataclass
class Paths:
    """Rutas configurables por variables de entorno.

    Cada variable se puede sobrescribir en docker-compose, en kubernetes
    o desde la línea de comandos. Los valores por defecto sirven para
    ejecuciones locales fuera de contenedor.
    """

    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("CARDIS_DATA_DIR", "./data"))
    )
    raw_train: Path = field(
        default_factory=lambda: Path(
            os.getenv("CARDIS_RAW_TRAIN", "./data/raw/cardis_train.csv")
        )
    )
    raw_test: Path = field(
        default_factory=lambda: Path(
            os.getenv("CARDIS_RAW_TEST", "./data/raw/cardis_test.csv")
        )
    )
    processed_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("CARDIS_PROCESSED_DIR", "./data/processed")
        )
    )
    model_dir: Path = field(
        default_factory=lambda: Path(os.getenv("CARDIS_MODEL_DIR", "./models"))
    )
    model_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("CARDIS_MODEL_PATH", "./models/cardis_lightgbm.joblib")
        )
    )
    calibrator_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("CARDIS_CALIBRATOR_PATH", "./models/cardis_calibrator.joblib")
        )
    )
    metadata_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("CARDIS_METADATA_PATH", "./models/cardis_metadata.json")
        )
    )

    def ensure(self) -> None:
        """Crea los directorios de salida si no existen."""
        for p in (self.processed_dir, self.model_dir):
            p.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# MinIO / S3
# -----------------------------------------------------------------------------
@dataclass
class StorageConfig:
    """Configuración del almacén de modelos (MinIO o sistema de ficheros).

    Si ``CARDIS_STORAGE_BACKEND=minio`` se activan las credenciales y se
    persisten los artefactos en un bucket. En cualquier otro caso se usa
    almacenamiento local (modo por defecto, útil para desarrollo).
    """

    backend: str = field(
        default_factory=lambda: os.getenv("CARDIS_STORAGE_BACKEND", "local")
    )
    endpoint: str = field(
        default_factory=lambda: os.getenv("MINIO_ENDPOINT", "minio:9000")
    )
    access_key: str = field(
        default_factory=lambda: os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    )
    secret_key: str = field(
        default_factory=lambda: os.getenv("MINIO_SECRET_KEY", "minioadmin")
    )
    bucket: str = field(
        default_factory=lambda: os.getenv("MINIO_BUCKET", "cardis-models")
    )
    secure: bool = field(
        default_factory=lambda: os.getenv("MINIO_SECURE", "false").lower() == "true"
    )


# -----------------------------------------------------------------------------
# MLflow
# -----------------------------------------------------------------------------
@dataclass
class MLflowConfig:
    """Configuración de MLflow para tracking de experimentos."""

    tracking_uri: Optional[str] = field(
        default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI")
    )
    experiment_name: str = field(
        default_factory=lambda: os.getenv(
            "MLFLOW_EXPERIMENT_NAME", "cardis-cardiovascular"
        )
    )

    @property
    def enabled(self) -> bool:
        return bool(self.tracking_uri)


# -----------------------------------------------------------------------------
# Hiperparámetros
# -----------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """Hiperparámetros optimos del modelo final (Hito 3).

    Se mantienen los valores convergentes detectados con Optuna en el
    cuaderno ``cardis_modelado.ipynb``. La clave ``scale_pos_weight`` se
    fija al ratio de la clase minoritaria (78/22) descrito en la memoria.
    """

    target_col: str = "riesgo_cv"
    test_size: float = 0.20
    random_state: int = 42
    n_splits: int = 5
    threshold: float = 0.59  # umbral optimo del Hito 3
    lgbm_params: dict = field(
        default_factory=lambda: {
            "objective": "binary",
            "metric": "binary_logloss",
            "n_estimators": 600,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": -1,
            "min_child_samples": 30,
            "reg_lambda": 0.99,
            "subsample": 0.83,
            "subsample_freq": 1,
            "colsample_bytree": 0.85,
            "scale_pos_weight": 3.56,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
        }
    )
