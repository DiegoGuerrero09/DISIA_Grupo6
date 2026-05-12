"""Logica de inferencia del sistema CARDIS.

Aisla la carga del modelo y el predicado para que tanto la API REST como
los scripts de batch reutilicen exactamente el mismo codigo. Cumple con
el principio del enunciado del Hito 4: separar entrenamiento e
inferencia en componentes independientes.

Uso por linea de comandos (modo batch)::

    python -m src.infer --input data/raw/cardis_test.csv \
        --output predictions.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd
from .config import ModelConfig, Paths, StorageConfig, setup_logging
from .features import FeatureBuilder, IsotonicCalibrator
from .model_store import ModelStore
from .schemas import PatientRecord, PredictResponse, PredictResult

logger = setup_logging()

YOUNG_AGE_THRESHOLD = 56  # sesgo conocido documentado en el Hito 3.


class CardisInferenceService:
    """Encapsula carga, validacion y predicado del modelo."""

    def __init__(
        self,
        model_path: Path,
        calibrator_path: Path,
        builder_path: Path,
        metadata_path: Optional[Path] = None,
        threshold: Optional[float] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.calibrator_path = Path(calibrator_path)
        self.builder_path = Path(builder_path)
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.threshold = threshold or ModelConfig().threshold

        self.model = None
        self.calibrator: Optional[IsotonicCalibrator] = None
        self.builder: Optional[FeatureBuilder] = None
        self.metadata: dict = {}

    # ------------------------------------------------------------- carga
    def load(self) -> "CardisInferenceService":
        store = ModelStore(StorageConfig())
        # Si las rutas locales no existen, intentamos descargarlas.
        for local in (self.model_path, self.calibrator_path, self.builder_path):
            if not local.exists():
                logger.info("Descargando %s desde el almacen", local.name)
                store.get(local.name, local)
        self.model = joblib.load(self.model_path)
        self.calibrator = joblib.load(self.calibrator_path)
        self.builder = FeatureBuilder.load(self.builder_path)

        if self.metadata_path and self.metadata_path.exists():
            with self.metadata_path.open("r", encoding="utf-8") as fh:
                self.metadata = json.load(fh)
            self.threshold = self.metadata.get("threshold", self.threshold)
        logger.info(
            "Modelo cargado (threshold=%.3f, %d features)",
            self.threshold,
            len(self.builder.feature_names_),
        )
        return self

    # ------------------------------------------------------------ predict
    def _ensure_loaded(self) -> None:
        if self.calibrator is None or self.builder is None:
            raise RuntimeError(
                "El servicio no se ha cargado: llama a .load() antes de predecir"
            )

    def predict_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        self._ensure_loaded()
        feats = self.builder.transform(df)  # type: ignore[union-attr]
        proba = self.calibrator.predict_proba(feats)[:, 1]
        return proba

    def predict(self, patients: Iterable[PatientRecord]) -> PredictResponse:
        self._ensure_loaded()
        records: List[dict] = [p.model_dump() for p in patients]
        df = pd.DataFrame.from_records(records)
        proba = self.predict_dataframe(df)
        results: List[PredictResult] = []
        for prob, age in zip(proba, df.get("edad", pd.Series(dtype=float))):
            label = int(prob >= self.threshold)
            results.append(
                PredictResult(
                    probability=float(prob),
                    label=label,
                    threshold=self.threshold,
                    age_warning=bool(age is not None and age < YOUNG_AGE_THRESHOLD),
                )
            )
        return PredictResponse(
            model_version=str(self.metadata.get("model_version", "1.0.0")),
            threshold=self.threshold,
            predictions=results,
        )


# -----------------------------------------------------------------------------
# Modo batch (CLI)
# -----------------------------------------------------------------------------
def run_batch(args: argparse.Namespace) -> None:
    paths = Paths()
    service = CardisInferenceService(
        model_path=args.model_path,
        calibrator_path=args.calibrator_path,
        builder_path=args.builder_path,
        metadata_path=args.metadata_path,
    ).load()

    df = pd.read_csv(args.input)
    proba = service.predict_dataframe(df)
    df["riesgo_proba"] = proba
    df["riesgo_label"] = (proba >= service.threshold).astype(int)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    logger.info("Predicciones escritas en %s", args.output)


def _build_parser() -> argparse.ArgumentParser:
    paths = Paths()
    parser = argparse.ArgumentParser(description="Inferencia batch de CARDIS")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=paths.model_path)
    parser.add_argument(
        "--calibrator-path", type=Path, default=paths.calibrator_path
    )
    parser.add_argument(
        "--builder-path",
        type=Path,
        default=paths.model_dir / "cardis_featurebuilder.joblib",
    )
    parser.add_argument("--metadata-path", type=Path, default=paths.metadata_path)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    run_batch(args)


if __name__ == "__main__":
    main()
