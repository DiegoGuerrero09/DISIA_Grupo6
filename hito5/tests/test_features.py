"""Pruebas unitarias del pipeline de features.

Verifican que las transformaciones del Hito 2 producen el mismo numero
de columnas en train y test, que no hay NaNs residuales y que las
keywords clinicas se detectan correctamente.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import FeatureBuilder, KEYWORDS, parse_actividad


def _sample_df(n: int = 50, *, with_target: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "edad": rng.integers(30, 85, n),
            "altura_cm": rng.normal(170, 8, n),
            "peso_kg": rng.normal(75, 12, n),
            "imc": rng.normal(26, 4, n),
            "presion_sistolica_1": rng.normal(125, 15, n),
            "presion_sistolica_2": rng.normal(125, 15, n),
            "presion_sistolica_3": rng.normal(125, 15, n),
            "colesterol_total": rng.normal(200, 35, n),
            "ldl": rng.normal(120, 30, n),
            "hdl": rng.normal(45, 8, n),
            "glucosa_ayunas": np.where(
                rng.random(n) > 0.4, rng.normal(95, 10, n), np.nan
            ),
            "fumador": rng.integers(0, 2, n),
            "antecedentes_familiares": rng.integers(0, 2, n),
            "hospital_origen": rng.choice(
                [f"HOSP-{i:03d}" for i in range(1, 21)], n
            ),
            "fecha_visita": pd.date_range("2020-01-01", periods=n, freq="D"),
            "notas_medicas": rng.choice(
                ["paciente normal", "diabetes tipo 2", "infarto previo", None],
                n,
            ),
            "actividad_fisica": rng.choice(
                ["2.5h", "poco", "5 horas", "1,5 horas"], n
            ),
            "id": np.arange(n),
            "talla_zapato": rng.integers(36, 46, n),
            "codigo_postal": rng.choice(["28001", "08001", "13001"], n),
        }
    )
    if with_target:
        df["riesgo_cv"] = rng.integers(0, 2, n)
    return df


def test_parse_actividad_extracts_numbers() -> None:
    assert parse_actividad("2.5h") == 2.5
    assert parse_actividad("3,5 horas") == 3.5
    assert parse_actividad("poco") == 0.0
    assert np.isnan(parse_actividad(None))


def test_feature_builder_fit_transform_idempotent() -> None:
    df = _sample_df(100)
    builder = FeatureBuilder()
    feats_train = builder.fit_transform(df)

    test_df = _sample_df(40, with_target=False)
    feats_test = builder.transform(test_df)

    assert feats_train.shape[1] == feats_test.shape[1]
    assert not feats_train.isna().any().any(), "Quedan NaNs tras transformar"
    assert not feats_test.isna().any().any(), "Quedan NaNs en test"
    assert all(f"kw_{k}" in feats_train.columns for k in KEYWORDS)


def test_feature_builder_handles_unknown_hospital() -> None:
    df = _sample_df(60)
    builder = FeatureBuilder().fit(df)
    test_df = _sample_df(10, with_target=False)
    test_df["hospital_origen"] = "HOSP-XYZ"  # nunca visto
    feats = builder.transform(test_df)
    # Todos los hospitales desconocidos deben recibir el mismo valor (el default
    # escalado); no podemos comparar contra target_encoding_default_ directamente
    # porque RobustScaler lo transforma.
    assert feats["hospital_te"].nunique() == 1


def test_dropped_columns_not_present() -> None:
    df = _sample_df(20)
    feats = FeatureBuilder().fit_transform(df)
    for col in ("id", "talla_zapato", "codigo_postal"):
        assert col not in feats.columns, f"La columna {col} debio descartarse"
