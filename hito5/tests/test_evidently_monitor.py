"""Pruebas unitarias de evidently_monitor (Hito 5 — Bloque 4).

Cubren:
- Que _download_current_window NO introduce la columna virtual 'dt'
  (problema documentado en hito5_notes.md B2).
- Que _update_gauges actualiza correctamente los cuatro Gauges.
- Que /v2/health/ready devuelve 503 antes del primer ciclo.
- Que /v2/health/live devuelve 200 siempre.
- Que _run_evidently produce el dict esperado con datos reales (sin mock
  de Evidently, para ejercitar la integración real del parser).

MinIO se mockea completamente; no se requiere servidor real.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Importaciones del módulo bajo test (aseguramos que evidently esté instalado)
# ---------------------------------------------------------------------------
pytest.importorskip("evidently", reason="evidently no instalado")

import src.evidently_monitor as em  # noqa: E402
from src.evidently_monitor import (  # noqa: E402
    _DRIFT_COLUMN,
    _DRIFT_DATASET,
    _DRIFT_SHARE,
    _LAST_RUN,
    _download_current_window,
    _run_evidently,
    _update_gauges,
    app,
)

# ---------------------------------------------------------------------------
# Constantes de test
# ---------------------------------------------------------------------------
FEATURE_COLS = em._HARDCODED_FEATURE_COLS  # las 36 conocidas


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_parquet_bytes(n: int = 15, include_dt_virtual: bool = False) -> bytes:
    """Crea un buffer Parquet en memoria con filas de inferencia simuladas.

    Los timestamps se generan relativos a ``datetime.now(UTC)`` para que
    siempre caigan dentro de cualquier ventana razonable de minutos.
    El fichero NO incluye la columna 'dt'; eso solo ocurre cuando PyArrow
    infiere el directorio Hive (lo que _download_current_window evita).
    """
    rng = np.random.default_rng(42)
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    rows = {
        "timestamp": [
            (now - timedelta(seconds=i * 10)).isoformat()
            for i in range(n)
        ],
        "model_version": ["1.0.0"] * n,
        "request_id": [f"req-{i:04d}" for i in range(n)],
        "worker_pid": [1234] * n,
        "patient_index": list(range(n)),
    }
    for col in FEATURE_COLS:
        rows[col] = rng.normal(0, 1, n).tolist()
    rows["probability"] = rng.random(n).tolist()
    rows["label"] = rng.integers(0, 2, n).tolist()
    rows["age_warning"] = [False] * n

    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Test 1: _download_current_window no produce columna 'dt'
# ---------------------------------------------------------------------------

def test_download_no_dt_column(tmp_path: Path) -> None:
    """_download_current_window no debe incluir la columna virtual 'dt'."""
    parquet_bytes = _make_parquet_bytes(n=15)

    # Guardar parquet en un fichero temporal para que fget_object lo "descargue"
    real_file = tmp_path / "14-00-1234.parquet"
    real_file.write_bytes(parquet_bytes)

    # Mock de objeto MinIO
    mock_obj = MagicMock()
    mock_obj.object_name = "dt=2026-05-17/14-00-1234.parquet"

    # Mock del cliente MinIO:
    # - list_objects devuelve el mock_obj
    # - fget_object copia el fichero real al path destino
    def fake_fget_object(bucket, object_name, dest_path):  # noqa: ARG001
        Path(dest_path).write_bytes(parquet_bytes)

    mock_client = MagicMock()
    mock_client.list_objects.return_value = [mock_obj]
    mock_client.fget_object.side_effect = fake_fget_object

    result = _download_current_window(
        client=mock_client,
        infer_bucket="cardis-inferences",
        window_min=60,  # ventana amplia para que los 15 registros caigan dentro
        feature_cols=FEATURE_COLS,
    )

    assert "dt" not in result.columns, "La columna virtual 'dt' no debe aparecer"
    assert len(result) == 15
    assert set(result.columns) == set(FEATURE_COLS)


# ---------------------------------------------------------------------------
# Test 2: _download_current_window con prefijo sin objetos no falla
# ---------------------------------------------------------------------------

def test_download_empty_prefix_returns_empty() -> None:
    """Si no hay objetos en el prefijo, devuelve DataFrame vacío sin error."""
    mock_client = MagicMock()
    mock_client.list_objects.return_value = []  # sin objetos

    result = _download_current_window(
        client=mock_client,
        infer_bucket="cardis-inferences",
        window_min=5,
        feature_cols=FEATURE_COLS,
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# Test 3: _update_gauges actualiza los cuatro Gauges correctamente
# ---------------------------------------------------------------------------

def test_update_gauges_with_drift() -> None:
    """_update_gauges debe reflejar el resultado de Evidently en los Gauges."""
    result = {
        "dataset_drift": True,
        "share": 0.5,
        "columns": {
            "edad": True,
            "imc": False,
            "ldl": True,
        },
    }
    _update_gauges(result)

    assert _DRIFT_DATASET._value.get() == 1.0  # type: ignore[attr-defined]
    assert _DRIFT_SHARE._value.get() == pytest.approx(0.5)  # type: ignore[attr-defined]
    assert _DRIFT_COLUMN.labels(feature="edad")._value.get() == 1.0  # type: ignore[attr-defined]
    assert _DRIFT_COLUMN.labels(feature="imc")._value.get() == 0.0  # type: ignore[attr-defined]
    assert _DRIFT_COLUMN.labels(feature="ldl")._value.get() == 1.0  # type: ignore[attr-defined]


def test_update_gauges_no_drift() -> None:
    """_update_gauges con dataset_drift=False pone el Gauge a 0."""
    result = {
        "dataset_drift": False,
        "share": 0.0,
        "columns": {"edad": False, "imc": False},
    }
    _update_gauges(result)

    assert _DRIFT_DATASET._value.get() == 0.0  # type: ignore[attr-defined]
    assert _DRIFT_SHARE._value.get() == pytest.approx(0.0)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Test 4: _run_evidently extrae el dict con la estructura correcta
# ---------------------------------------------------------------------------

def test_run_evidently_with_real_drift() -> None:
    """_run_evidently con shift intencional detecta drift y produce dict válido."""
    rng = np.random.default_rng(42)
    cols = FEATURE_COLS[:5]  # subset para rapidez
    ref = pd.DataFrame(rng.normal(0, 1, (80, len(cols))), columns=cols)
    # shift marcado: media +5 sobre la distribución de referencia
    cur = pd.DataFrame(rng.normal(5, 1, (30, len(cols))), columns=cols)

    result = _run_evidently(ref, cur)

    assert isinstance(result, dict)
    assert "dataset_drift" in result
    assert "share" in result
    assert "columns" in result
    assert isinstance(result["dataset_drift"], bool)
    assert 0.0 <= result["share"] <= 1.0
    assert all(isinstance(v, bool) for v in result["columns"].values())
    # Con shift=5 todos deben tener drift
    assert result["dataset_drift"] is True
    assert result["share"] == pytest.approx(1.0)


def test_run_evidently_no_drift() -> None:
    """_run_evidently con datos idénticos no detecta drift."""
    rng = np.random.default_rng(99)
    cols = FEATURE_COLS[:5]
    ref = pd.DataFrame(rng.normal(0, 1, (200, len(cols))), columns=cols)
    cur = pd.DataFrame(rng.normal(0, 1, (50, len(cols))), columns=cols)

    result = _run_evidently(ref, cur)

    assert isinstance(result["dataset_drift"], bool)
    # Con distribuciones idénticas, la mayoría de features no deben derivar
    assert result["share"] < 0.5


# ---------------------------------------------------------------------------
# Test 5: endpoints HTTP
# ---------------------------------------------------------------------------

def test_ready_returns_503_before_cycle() -> None:
    """GET /v2/health/ready → 503 cuando _ready es False."""
    original = em._ready
    try:
        em._ready = False
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/v2/health/ready")
        assert resp.status_code == 503
    finally:
        em._ready = original


def test_ready_returns_200_after_cycle() -> None:
    """GET /v2/health/ready → 200 cuando _ready es True."""
    original = em._ready
    try:
        em._ready = True
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/v2/health/ready")
        assert resp.status_code == 200
    finally:
        em._ready = original


def test_live_always_200() -> None:
    """GET /v2/health/live → 200 siempre."""
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get("/v2/health/live")
    assert resp.status_code == 200


def test_metrics_endpoint_returns_text() -> None:
    """GET /metrics devuelve texto plano con métricas Prometheus."""
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    assert "cardis_drift_last_run_timestamp_seconds" in resp.text
