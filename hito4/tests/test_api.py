"""Pruebas de integracion de la API.

Comprueban los contratos de los endpoints publicos sin necesidad de
levantar el contenedor. Para que funcione es necesario haber entrenado
previamente el modelo (``python -m src.train``).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

# Si no hay modelo entrenado los tests se saltan automaticamente para
# evitar fallos en CI sin artefactos.
if not Path(os.getenv("CARDIS_MODEL_PATH", "./models/cardis_lightgbm.joblib")).exists():
    pytest.skip("No hay modelo entrenado en disco", allow_module_level=True)


def _client():
    from src.api import app
    return TestClient(app)


def test_health_live() -> None:
    with _client() as client:
        response = client.get("/v2/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_high_level_endpoint() -> None:
    payload = {
        "patients": [
            {
                "edad": 45,
                "altura_cm": 172,
                "peso_kg": 78,
                "imc": 26.4,
                "presion_sistolica_1": 132,
                "presion_sistolica_2": 130,
                "presion_sistolica_3": 134,
                "colesterol_total": 220,
                "ldl": 130,
                "hdl": 48,
                "glucosa_ayunas": 99,
                "fumador": 0,
                "antecedentes_familiares": 1,
                "hospital_origen": "HOSP-001",
                "fecha_visita": "2024-09-12",
                "notas_medicas": "paciente con diabetes tipo 2 controlada",
                "actividad_fisica": "2 horas semanales",
            }
        ]
    }
    with _client() as client:
        response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert 0.0 <= data["predictions"][0]["probability"] <= 1.0
    assert data["predictions"][0]["age_warning"] is True  # 45 < 56
