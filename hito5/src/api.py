"""API REST de inferencia del sistema CARDIS (FastAPI).

Compatible con el Open Inference Protocol (KServe v2) y con un endpoint
de alto nivel (``/predict``) pensado para la interfaz clinica del Hito 1.

Endpoints expuestos:

* ``GET  /v2/health/live``  -> liveness probe (Kubernetes).
* ``GET  /v2/health/ready`` -> readiness probe.
* ``GET  /v2/models/{name}`` -> metadatos del modelo.
* ``POST /v2/models/{name}/infer`` -> inferencia OIP.
* ``POST /predict`` -> inferencia humana (JSON tabular).
* ``GET  /metrics`` -> metricas Prometheus.

La eleccion de FastAPI se ha justificado en la memoria del Hito 4: ofrece
validacion automatica via Pydantic, generacion de OpenAPI, soporte
asincrono y un overhead bajo por peticion frente a Flask.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from .config import Paths, setup_logging
from .infer import CardisInferenceService
from .schemas import (
    HealthResponse,
    InferenceRequest,
    InferenceResponse,
    InferenceTensor,
    ModelMetadata,
    PredictRequest,
    PredictResponse,
)

logger = setup_logging()
SERVICE: CardisInferenceService | None = None
START_TIME: float = time.time()
REQUEST_COUNTER: dict[str, int] = {"total": 0, "errors": 0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo una sola vez al arrancar el contenedor."""
    global SERVICE
    paths = Paths()
    builder_path = paths.model_dir / "cardis_featurebuilder.joblib"
    SERVICE = CardisInferenceService(
        model_path=paths.model_path,
        calibrator_path=paths.calibrator_path,
        builder_path=builder_path,
        metadata_path=paths.metadata_path,
    )
    try:
        SERVICE.load()
        logger.info("API CARDIS lista para servir")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error al cargar el modelo: %s", exc)
    yield
    logger.info("Apagando API CARDIS")


app = FastAPI(
    title="CARDIS Inference API",
    description=(
        "Servicio de apoyo a la decision clinica para la prediccion "
        "interpretable del riesgo cardiovascular. Cumple con el Open "
        "Inference Protocol v2."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS abierto a la red interna del cluster (en produccion se restringe
# a los origenes autorizados de la interfaz clinica).
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CARDIS_CORS_ALLOW", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _service() -> CardisInferenceService:
    if SERVICE is None or SERVICE.calibrator is None:
        raise HTTPException(
            status_code=503, detail="El modelo aun no esta listo"
        )
    return SERVICE


# -----------------------------------------------------------------------------
# Health checks
# -----------------------------------------------------------------------------
@app.get("/v2/health/live", response_model=HealthResponse)
async def health_live() -> HealthResponse:
    return HealthResponse(model_loaded=SERVICE is not None and SERVICE.calibrator is not None)


@app.get("/v2/health/ready", response_model=HealthResponse)
async def health_ready() -> HealthResponse:
    if SERVICE is None or SERVICE.calibrator is None:
        raise HTTPException(status_code=503, detail="No listo")
    return HealthResponse(model_loaded=True)


# -----------------------------------------------------------------------------
# Open Inference Protocol
# -----------------------------------------------------------------------------
@app.get("/v2/models/{model_name}", response_model=ModelMetadata)
async def get_metadata(model_name: str) -> ModelMetadata:
    service = _service()
    if model_name not in {"cardis-lightgbm", "cardis"}:
        raise HTTPException(status_code=404, detail="Modelo desconocido")
    return ModelMetadata(
        name="cardis-lightgbm",
        version="1.0.0",
        platform="lightgbm",
        threshold=service.threshold,
        feature_names=service.builder.feature_names_,  # type: ignore[union-attr]
    )


@app.post("/v2/models/{model_name}/infer", response_model=InferenceResponse)
async def infer_oip(model_name: str, request: InferenceRequest) -> InferenceResponse:
    service = _service()
    REQUEST_COUNTER["total"] += 1
    try:
        if model_name not in {"cardis-lightgbm", "cardis"}:
            raise HTTPException(status_code=404, detail="Modelo desconocido")
        tensor = next(
            (t for t in request.inputs if t.name == "patients_json"),
            request.inputs[0],
        )
        # Esperamos un tensor cuya carga sea una lista de dicts JSON
        import pandas as pd

        df = pd.DataFrame.from_records(tensor.data)
        proba = service.predict_dataframe(df)
        labels = (proba >= service.threshold).astype(int)
        outputs = [
            InferenceTensor(
                name="probability",
                shape=[len(proba), 1],
                datatype="FP32",
                data=proba.tolist(),
            ),
            InferenceTensor(
                name="label",
                shape=[len(labels), 1],
                datatype="INT32",
                data=labels.tolist(),
            ),
        ]
        return InferenceResponse(
            id=request.id,
            parameters={"threshold": service.threshold},
            outputs=outputs,
        )
    except HTTPException:
        REQUEST_COUNTER["errors"] += 1
        raise
    except Exception as exc:  # noqa: BLE001
        REQUEST_COUNTER["errors"] += 1
        logger.exception("Error en /v2/infer: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# -----------------------------------------------------------------------------
# Endpoint humano
# -----------------------------------------------------------------------------
@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    service = _service()
    REQUEST_COUNTER["total"] += 1
    try:
        return service.predict(request.patients)
    except Exception as exc:  # noqa: BLE001
        REQUEST_COUNTER["errors"] += 1
        logger.exception("Error en /predict: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# -----------------------------------------------------------------------------
# Metrics (Prometheus, formato texto)
# -----------------------------------------------------------------------------
@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> str:
    uptime = time.time() - START_TIME
    return (
        f"# HELP cardis_uptime_seconds Tiempo desde el arranque del servicio\n"
        f"# TYPE cardis_uptime_seconds counter\n"
        f"cardis_uptime_seconds {uptime:.2f}\n"
        f"# HELP cardis_requests_total Numero total de peticiones servidas\n"
        f"# TYPE cardis_requests_total counter\n"
        f"cardis_requests_total {REQUEST_COUNTER['total']}\n"
        f"# HELP cardis_request_errors_total Numero de errores devueltos\n"
        f"# TYPE cardis_request_errors_total counter\n"
        f"cardis_request_errors_total {REQUEST_COUNTER['errors']}\n"
    )


# -----------------------------------------------------------------------------
# Root
# -----------------------------------------------------------------------------
@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse(
        {
            "name": "CARDIS Inference API",
            "version": "1.0.0",
            "docs": "/docs",
            "openapi": "/openapi.json",
            "health": "/v2/health/ready",
        }
    )
