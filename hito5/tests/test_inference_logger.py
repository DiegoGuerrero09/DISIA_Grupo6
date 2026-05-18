"""Pruebas unitarias de InferenceLogger (Hito 5 — Bloque 2).

Verifican el comportamiento del buffer, las condiciones de flush y que
los errores de MinIO no propagan excepciones.  MinIO se mockea
completamente: no se necesita ningún servidor real para ejecutar estos
tests.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from src.inference_logger import InferenceLogger, InferenceRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(patient_index: int = 0) -> InferenceRecord:
    """Construye un InferenceRecord mínimo válido para tests."""
    return {
        "timestamp": "2024-01-01T00:00:00+00:00",
        "model_version": "1.0.0",
        "request_id": "test-uuid-0001",
        "worker_pid": 1234,
        "patient_index": patient_index,
        "edad": 60.0,
        "altura_cm": 175.0,
        "peso_kg": 80.0,
        "imc": 26.1,
        "presion_sistolica_1": 130.0,
        "presion_sistolica_2": 128.0,
        "presion_sistolica_3": 132.0,
        "colesterol_total": 200.0,
        "ldl": 120.0,
        "hdl": 45.0,
        "glucosa_ayunas": 95.0,
        "fumador": 0.0,
        "antecedentes_familiares": 0.0,
        "glucosa_missing": 0.0,
        "notas_missing": 0.0,
        "actividad_missing": 0.0,
        "hospital_te": 1.0,
        "horas_actividad": 4.0,
        "anio_visita": 2023.0,
        "mes_visita": 6.0,
        "dia_semana": 2.0,
        "hora_visita": 10.0,
        "es_fin_semana": 0.0,
        "kw_diabetes": 0.0,
        "kw_infarto": 0.0,
        "kw_coronario": 0.0,
        "kw_mareo": 0.0,
        "kw_pecho": 0.0,
        "kw_fatiga": 0.0,
        "kw_dolor": 0.0,
        "kw_sedentarismo": 0.0,
        "kw_angina": 0.0,
        "kw_fumador": 0.0,
        "presion_sistolica_media": 130.0,
        "presion_sistolica_mediana": 130.0,
        "presion_sistolica_std": 2.0,
        "probability": 0.35,
        "label": 0,
        "age_warning": False,
    }  # type: ignore[return-value]


def _make_logger(**kwargs) -> InferenceLogger:
    """Crea un InferenceLogger con parámetros de test (sin conexiones reales)."""
    defaults = {
        "bucket": "cardis-inferences",
        "endpoint": "localhost:9000",
        "access_key": "minioadmin",
        "secret_key": "minioadmin",
        "flush_size": 50,
        "flush_interval_s": 60.0,
        "min_flush_size": 5,
        "max_flush_wait_s": 300.0,
    }
    defaults.update(kwargs)
    return InferenceLogger(**defaults)


# ---------------------------------------------------------------------------
# Test 1: log() añade al buffer sin flushear
# ---------------------------------------------------------------------------

def test_log_appends_to_buffer() -> None:
    """log() añade el registro al buffer y fija el timestamp del primer registro."""
    logger = _make_logger()

    assert len(logger._buffer) == 0
    assert logger._first_record_ts is None

    logger.log(_make_record(patient_index=0))

    assert len(logger._buffer) == 1
    assert logger._first_record_ts is not None

    # Un segundo log no sobreescribe el primer timestamp
    first_ts = logger._first_record_ts
    logger.log(_make_record(patient_index=1))
    assert len(logger._buffer) == 2
    assert logger._first_record_ts == first_ts


# ---------------------------------------------------------------------------
# Test 2: flush() con registros llama a Minio.fput_object y vacía el buffer
# ---------------------------------------------------------------------------

def test_flush_uploads_parquet_and_clears_buffer() -> None:
    """flush() sube un Parquet a MinIO y deja el buffer vacío."""
    logger = _make_logger()
    for i in range(3):
        logger.log(_make_record(patient_index=i))

    assert len(logger._buffer) == 3

    with patch("src.inference_logger.Minio") as mock_minio_cls:
        mock_client = MagicMock()
        mock_minio_cls.return_value = mock_client

        asyncio.run(logger.flush())

    assert len(logger._buffer) == 0
    mock_client.fput_object.assert_called_once()
    # El bucket debe ser el correcto
    call_args = mock_client.fput_object.call_args
    assert call_args[0][0] == "cardis-inferences"


# ---------------------------------------------------------------------------
# Test 3: condición de flush por tiempo con min_flush_size registros
# ---------------------------------------------------------------------------

def test_flush_by_time_condition() -> None:
    """_should_flush devuelve True con >= min_flush_size registros y tiempo suficiente."""
    logger = _make_logger(min_flush_size=5, flush_interval_s=60.0, flush_size=50)

    # Justo en el límite: 5 registros, 60 segundos transcurridos
    assert logger._should_flush(n=5, elapsed=60.0) is True
    # Por encima del límite
    assert logger._should_flush(n=10, elapsed=120.0) is True


# ---------------------------------------------------------------------------
# Test 4: con menos de min_flush_size registros NO se flushea por tiempo
# ---------------------------------------------------------------------------

def test_flush_below_min_size_does_not_trigger() -> None:
    """_should_flush devuelve False con < min_flush_size y tiempo transcurrido."""
    logger = _make_logger(
        min_flush_size=5,
        flush_interval_s=60.0,
        flush_size=50,
        max_flush_wait_s=300.0,
    )

    # 4 registros (< min_flush_size=5), tiempo > flush_interval_s pero < max_flush_wait
    assert logger._should_flush(n=4, elapsed=90.0) is False
    # Límite estricto: exactamente min_flush_size-1 registros
    assert logger._should_flush(n=4, elapsed=299.0) is False


# ---------------------------------------------------------------------------
# Test 5: max_flush_wait_s fuerza el flush aunque haya < min_flush_size registros
# ---------------------------------------------------------------------------

def test_flush_max_wait_forces_flush() -> None:
    """_should_flush devuelve True con >= 1 registro y elapsed >= max_flush_wait_s."""
    logger = _make_logger(
        min_flush_size=5,
        flush_interval_s=60.0,
        flush_size=50,
        max_flush_wait_s=300.0,
    )

    # 1 solo registro pero han pasado 300 s
    assert logger._should_flush(n=1, elapsed=300.0) is True
    # Justo por debajo: NO debe flushear
    assert logger._should_flush(n=1, elapsed=299.9) is False


# ---------------------------------------------------------------------------
# Test 6: error de MinIO no propaga excepción
# ---------------------------------------------------------------------------

def test_minio_error_does_not_propagate() -> None:
    """Si MinIO lanza una excepción, flush() no la propaga."""
    logger = _make_logger()
    for i in range(3):
        logger.log(_make_record(patient_index=i))

    with patch("src.inference_logger.Minio") as mock_minio_cls:
        mock_client = MagicMock()
        mock_client.fput_object.side_effect = Exception("conexión rechazada")
        mock_minio_cls.return_value = mock_client

        # No debe lanzar excepción
        asyncio.run(logger.flush())

    # El buffer se vacía antes del error (los registros ya se consumieron)
    assert len(logger._buffer) == 0


# ---------------------------------------------------------------------------
# Test 7: stop() vacía el buffer aunque tenga < min_flush_size registros
# ---------------------------------------------------------------------------

def test_stop_flushes_remaining_records() -> None:
    """stop() hace un flush final aunque haya menos registros que min_flush_size."""
    logger = _make_logger(min_flush_size=5)
    # Solo 2 registros (< min_flush_size=5)
    logger.log(_make_record(patient_index=0))
    logger.log(_make_record(patient_index=1))

    assert len(logger._buffer) == 2

    with patch("src.inference_logger.Minio") as mock_minio_cls:
        mock_client = MagicMock()
        mock_minio_cls.return_value = mock_client

        asyncio.run(logger.stop())

    assert len(logger._buffer) == 0
    mock_client.fput_object.assert_called_once()
