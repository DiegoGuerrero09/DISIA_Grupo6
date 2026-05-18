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
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import requests

# config DEBE importarse antes que prometheus_client: su side-effect
# (os.environ.setdefault de PROMETHEUS_MULTIPROC_DIR) debe ejecutarse
# antes de que prometheus_client detecte el modo de operación.
from .config import Paths, StorageConfig, setup_logging  # noqa: E402

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client.multiprocess import MultiProcessCollector
from starlette.middleware.base import BaseHTTPMiddleware

from .infer import CardisInferenceService
from .inference_logger import InferenceLogger, InferenceRecord
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
INFERENCE_LOGGER: InferenceLogger | None = None

# ---------------------------------------------------------------------------
# Métricas Prometheus (8 métricas del Bloque 1)
# ---------------------------------------------------------------------------
_HTTP_REQUESTS = Counter(
    "cardis_http_requests_total",
    "Número de peticiones HTTP atendidas",
    ["endpoint", "method", "status_code"],
)
_HTTP_LATENCY = Histogram(
    "cardis_http_request_duration_seconds",
    "Duración de peticiones HTTP en segundos",
    ["endpoint"],
)
_INFER_LATENCY = Histogram(
    "cardis_inference_duration_seconds",
    "Duración de la inferencia del modelo en segundos",
)
_IN_FLIGHT = Gauge(
    "cardis_in_flight_requests",
    "Peticiones HTTP actualmente en vuelo",
    multiprocess_mode="livesum",
)
_PREDICTIONS = Counter(
    "cardis_predictions_total",
    "Número de predicciones emitidas por etiqueta",
    ["label_predicho"],
)
_PRED_PROBA = Histogram(
    "cardis_prediction_probability",
    "Distribución de probabilidades de riesgo predichas",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
_AGE_WARNINGS = Counter(
    "cardis_age_warnings_total",
    "Número de avisos de edad joven emitidos",
)
_MODEL_INFO = Gauge(
    "cardis_model_info",
    "Información del modelo cargado (siempre 1)",
    ["version"],
)
_SHADOW_AGREEMENT = Gauge(
    "cardis_shadow_agreement", 
    "Porcentaje de coincidencia entre modelo oficial y shadow",
    ["model_version_shadow"]
)


class _MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware ASGI para métricas operativas de HTTP."""

    async def dispatch(self, request: Request, call_next) -> Response:
        endpoint = request.url.path
        method = request.method
        _IN_FLIGHT.inc()
        start = time.perf_counter()
        try:
            response: Response = await call_next(request)
            status = str(response.status_code)
        except Exception:
            status = "500"
            raise
        finally:
            elapsed = time.perf_counter() - start
            _IN_FLIGHT.dec()
            _HTTP_REQUESTS.labels(
                endpoint=endpoint, method=method, status_code=status
            ).inc()
            _HTTP_LATENCY.labels(endpoint=endpoint).observe(elapsed)
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo una sola vez al arrancar el contenedor."""
    global SERVICE, INFERENCE_LOGGER
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
        version = SERVICE.metadata.get("model_version", "1.0.0")
        _MODEL_INFO.labels(version=version).set(1)
        storage = StorageConfig()
        INFERENCE_LOGGER = InferenceLogger(
            bucket=os.getenv("CARDIS_INFERENCES_BUCKET", "cardis-inferences"),
            endpoint=storage.endpoint,
            access_key=storage.access_key,
            secret_key=storage.secret_key,
            secure=storage.secure,
        )
        await INFERENCE_LOGGER.start_background_task()
        logger.info("API CARDIS lista para servir")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error al cargar el modelo: %s", exc)
    yield
    if INFERENCE_LOGGER is not None:
        await INFERENCE_LOGGER.stop()
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
app.add_middleware(_MetricsMiddleware)


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
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error en /v2/infer: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# -----------------------------------------------------------------------------
# Endpoint humano
# -----------------------------------------------------------------------------
@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    service = _service()
    request_id = str(uuid4())
    try:
        start = time.perf_counter()
        response, feat_df = service.predict_with_features(request.patients)
        _INFER_LATENCY.observe(time.perf_counter() - start)

        SHADOW_URL = "http://cardis-inferer-shadow.cardis.svc.cluster.local/predict"

        try:
            shadow_res = requests.post(
                SHADOW_URL,
                json=request.dict(),
                timeout=0.2
            )
            shadow_res.raise_for_status()
            shadow_data = shadow_res.json()

            main_labels = [p.label for p in response.predictions]
            shadow_labels = [p["label"] for p in shadow_data.get("predictions", [])]

            if not main_labels or not shadow_labels:
                agreement = 0.0
            else:
                matches = sum(
                    1 for m, s in zip(main_labels, shadow_labels) if m == s
                )
                agreement = matches / min(len(main_labels), len(shadow_labels))

            _SHADOW_AGREEMENT.labels(
                model_version_shadow="1.1.0-retrained"
            ).set(agreement)
        except Exception as e:
            logger.warning("Shadow comparison failed: %s", e)

        for pred in response.predictions:
            _PREDICTIONS.labels(label_predicho=str(pred.label)).inc()
            _PRED_PROBA.observe(pred.probability)
            if pred.age_warning:
                _AGE_WARNINGS.inc()
        response.request_id = request_id
        if INFERENCE_LOGGER is not None:
            try:
                ts = datetime.now(timezone.utc).isoformat()
                model_version = response.model_version
                pid = os.getpid()
                for idx, (pred, (_, feat_row)) in enumerate(
                    zip(response.predictions, feat_df.iterrows())
                ):
                    record: InferenceRecord = {
                        "timestamp": ts,
                        "model_version": model_version,
                        "request_id": request_id,
                        "worker_pid": pid,
                        "patient_index": idx,
                        **feat_row.to_dict(),
                        "probability": pred.probability,
                        "label": pred.label,
                        "age_warning": pred.age_warning,
                    }
                    INFERENCE_LOGGER.log(record)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "InferenceLogger: error al construir registro, omitiendo"
                )
        return response
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error en /predict: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# -----------------------------------------------------------------------------
# Metrics (Prometheus, formato texto)
# -----------------------------------------------------------------------------
@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    """Expone métricas Prometheus en formato texto (modo multiprocess)."""
    registry = CollectorRegistry()
    MultiProcessCollector(registry)
    data = generate_latest(registry)
    return PlainTextResponse(content=data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)


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
