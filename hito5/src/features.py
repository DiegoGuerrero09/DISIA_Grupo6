"""Ingenieria de caracteristicas del sistema CARDIS.

Replica el pipeline de preprocesamiento documentado en el Hito 2:

* Imputacion diferencial (mediana, moda e indicador binario de missing).
* Target Encoding sobre ``hospital_origen``.
* Extraccion de componentes temporales de ``fecha_visita``.
* Parseo del campo de texto ``actividad_fisica`` para obtener horas.
* Generacion de keywords clinicas sobre ``notas_medicas``.
* Agregados de las tres mediciones de presion sistolica.
* RobustScaler sobre las variables numericas.

El modulo expone una clase ``FeatureBuilder`` con metodos ``fit`` y
``transform`` para que tanto el contenedor de entrenamiento como el de
inferencia consuman exactamente la misma logica.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


class IsotonicCalibrator:
    """Calibrador isotónico para modelos pre-entrenados.

    Reemplaza CalibratedClassifierCV(cv='prefit') que fue eliminado en
    scikit-learn >= 1.6. Almacena el modelo base y la regresión isotónica
    por separado y expone predict_proba() para ser intercambiable.
    """

    def __init__(self, estimator: object, iso: object) -> None:
        self.estimator = estimator
        self._iso = iso

    def predict_proba(self, X: "pd.DataFrame") -> np.ndarray:
        raw: np.ndarray = self.estimator.predict_proba(X)[:, 1]  # type: ignore[union-attr]
        cal = np.clip(self._iso.predict(raw), 0.0, 1.0)  # type: ignore[union-attr]
        return np.column_stack([1.0 - cal, cal])

from .config import setup_logging

logger = setup_logging()

# Listas de columnas conocidas a partir del EDA documentado en el Hito 2.
NUMERIC_COLS: List[str] = [
    "edad",
    "altura_cm",
    "peso_kg",
    "imc",
    "presion_sistolica_1",
    "presion_sistolica_2",
    "presion_sistolica_3",
    "colesterol_total",
    "ldl",
    "hdl",
    "glucosa_ayunas",
]

BINARY_COLS: List[str] = ["fumador", "antecedentes_familiares"]

DROP_COLS: List[str] = ["id", "talla_zapato", "codigo_postal"]
# Nota: codigo_postal se descarta como recomendacion del Hito 3 para
# evitar discriminacion geografica indirecta documentada en SHAP.

# Keywords clinicas validadas en el EDA del Hito 2.
KEYWORDS: List[str] = [
    "diabetes",
    "infarto",
    "coronario",
    "mareo",
    "pecho",
    "fatiga",
    "dolor",
    "sedentarismo",
    "angina",
    "fumador",
]


# -----------------------------------------------------------------------------
# Funciones puras de transformacion
# -----------------------------------------------------------------------------
def parse_actividad(value: object) -> float:
    """Extrae las horas semanales del campo de texto libre.

    Acepta cadenas como ``"2.5h"``, ``"poco"``, ``"3,5 horas"``, etc.
    Devuelve ``np.nan`` si no se encuentra un numero interpretable.
    """
    if pd.isna(value):
        return np.nan
    text = str(value).lower().replace(",", ".")
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if match:
        return float(match.group(1))
    if "poco" in text or "nada" in text:
        return 0.0
    return np.nan


def extract_temporal(series: pd.Series) -> pd.DataFrame:
    """Convierte ``fecha_visita`` en un bloque de variables temporales."""
    dt = pd.to_datetime(series, errors="coerce")
    out = pd.DataFrame(
        {
            "anio_visita": dt.dt.year,
            "mes_visita": dt.dt.month,
            "dia_semana": dt.dt.dayofweek,
            "hora_visita": dt.dt.hour,
            "es_fin_semana": (dt.dt.dayofweek >= 5).astype("Int64"),
        }
    )
    return out


def extract_keywords(series: pd.Series) -> pd.DataFrame:
    """Genera columnas binarias kw_* a partir de ``notas_medicas``."""
    text = series.fillna("").astype(str).str.lower()
    return pd.DataFrame({f"kw_{kw}": text.str.contains(kw).astype("Int64") for kw in KEYWORDS})


def aggregate_pressure(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula media, mediana y desviacion tipica de la presion sistolica."""
    cols = ["presion_sistolica_1", "presion_sistolica_2", "presion_sistolica_3"]
    sub = df[cols].astype(float)
    return pd.DataFrame(
        {
            "presion_sistolica_media": sub.mean(axis=1),
            "presion_sistolica_mediana": sub.median(axis=1),
            "presion_sistolica_std": sub.std(axis=1),
        }
    )


# -----------------------------------------------------------------------------
# FeatureBuilder
# -----------------------------------------------------------------------------
@dataclass
class FeatureBuilder:
    """Pipeline reutilizable de ingenieria de caracteristicas.

    Se entrena en el contenedor de entrenamiento (``fit``) y se serializa
    junto con el modelo. El contenedor de inferencia carga el objeto y
    aplica ``transform`` sobre cada peticion entrante, garantizando que
    no exista ``train/serve skew``.
    """

    target_col: str = "riesgo_cv"
    target_encoding_: Dict[str, float] = field(default_factory=dict)
    target_encoding_default_: float = 0.0
    imputation_: Dict[str, float] = field(default_factory=dict)
    smoker_mode_: int = 0
    scaler_: Optional[RobustScaler] = None
    numeric_features_: List[str] = field(default_factory=list)
    feature_names_: List[str] = field(default_factory=list)
    fitted_: bool = False

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        """Aprende parametros de imputacion, encoding y escalado."""
        if self.target_col not in df.columns:
            raise ValueError(
                f"La columna objetivo '{self.target_col}' no esta en el dataframe"
            )
        logger.info("Entrenando FeatureBuilder con %d filas", len(df))

        # 1) Target encoding del hospital
        self.target_encoding_ = (
            df.groupby("hospital_origen")[self.target_col].mean().to_dict()
        )
        self.target_encoding_default_ = float(df[self.target_col].mean())

        # 2) Aprender medianas / modas
        for col in NUMERIC_COLS:
            self.imputation_[col] = float(df[col].median())
        self.smoker_mode_ = int(df["fumador"].mode().iloc[0]) if "fumador" in df else 0

        # 3) Construir features para entrenar el scaler
        feats = self._compute_features(df, fit_phase=True)
        self.numeric_features_ = [c for c in feats.columns if feats[c].dtype.kind in "fi"]
        self.scaler_ = RobustScaler().fit(feats[self.numeric_features_])
        self.feature_names_ = list(feats.columns)
        self.fitted_ = True
        logger.info(
            "FeatureBuilder entrenado: %d features finales", len(self.feature_names_)
        )
        return self

    # -------------------------------------------------------------- transform
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("FeatureBuilder no ha sido entrenado todavia")
        feats = self._compute_features(df, fit_phase=False)
        # Reordenar columnas para que coincidan con el entrenamiento
        for col in self.feature_names_:
            if col not in feats.columns:
                feats[col] = 0
        feats = feats[self.feature_names_].copy()
        feats[self.numeric_features_] = self.scaler_.transform(
            feats[self.numeric_features_]
        )
        return feats

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # ------------------------------------------------------------- internals
    def _compute_features(self, df: pd.DataFrame, fit_phase: bool) -> pd.DataFrame:
        df = df.copy()
        # --- Imputaciones
        df["glucosa_missing"] = df["glucosa_ayunas"].isna().astype("Int64")
        df["notas_missing"] = df["notas_medicas"].isna().astype("Int64") if "notas_medicas" in df else 0
        df["actividad_missing"] = (
            df["actividad_fisica"].isna().astype("Int64")
            if "actividad_fisica" in df
            else 0
        )
        for col, value in self.imputation_.items():
            if col in df.columns:
                df[col] = df[col].fillna(value)
        if "fumador" in df.columns:
            df["fumador"] = df["fumador"].fillna(self.smoker_mode_).astype(int)
        if "antecedentes_familiares" in df.columns:
            df["antecedentes_familiares"] = df["antecedentes_familiares"].fillna(0).astype(int)

        # --- Hospital -> Target Encoding
        if "hospital_origen" in df.columns:
            df["hospital_te"] = df["hospital_origen"].map(self.target_encoding_).fillna(
                self.target_encoding_default_
            )
        else:
            df["hospital_te"] = self.target_encoding_default_

        # --- Variables temporales y derivadas de texto
        if "fecha_visita" in df.columns:
            temporal = extract_temporal(df["fecha_visita"])
        else:
            temporal = pd.DataFrame(index=df.index)

        if "actividad_fisica" in df.columns:
            df["horas_actividad"] = df["actividad_fisica"].apply(parse_actividad)
            df["horas_actividad"] = df["horas_actividad"].fillna(
                df["horas_actividad"].median()
                if df["horas_actividad"].notna().any()
                else 0.0
            )
        else:
            df["horas_actividad"] = 0.0

        if "notas_medicas" in df.columns:
            kw = extract_keywords(df["notas_medicas"])
        else:
            kw = pd.DataFrame({f"kw_{k}": 0 for k in KEYWORDS}, index=df.index)

        pressure = aggregate_pressure(df)

        # --- Composicion final
        keep_numeric = [c for c in NUMERIC_COLS if c in df.columns]
        out = pd.concat(
            [
                df[keep_numeric],
                df[BINARY_COLS] if all(c in df.columns for c in BINARY_COLS) else pd.DataFrame(index=df.index),
                df[["hospital_te", "horas_actividad", "glucosa_missing",
                    "notas_missing", "actividad_missing"]],
                temporal.reset_index(drop=True).set_index(df.index)
                if not temporal.empty
                else pd.DataFrame(index=df.index),
                kw,
                pressure,
            ],
            axis=1,
        )
        # Eliminar columnas no informativas si todavia estan presentes
        for col in DROP_COLS:
            if col in out.columns:
                out = out.drop(columns=col)
        # Forzar tipos numericos coherentes
        out = out.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return out

    # ---------------------------------------------------------------- I/O
    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("FeatureBuilder serializado en %s", path)
        return path

    @classmethod
    def load(cls, path: Path) -> "FeatureBuilder":
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError("El artefacto cargado no es un FeatureBuilder")
        return obj


def split_xy(df: pd.DataFrame, target_col: str = "riesgo_cv") -> Tuple[pd.DataFrame, pd.Series]:
    """Separa features y target, dejando el target intacto."""
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])
    return X, y


def write_feature_metadata(builder: FeatureBuilder, path: Path) -> Path:
    """Persiste el listado de features para auditoria/reproducibilidad."""
    payload = {
        "n_features": len(builder.feature_names_),
        "feature_names": builder.feature_names_,
        "numeric_features": builder.numeric_features_,
        "keywords": KEYWORDS,
        "target_encoding_default": builder.target_encoding_default_,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return path
