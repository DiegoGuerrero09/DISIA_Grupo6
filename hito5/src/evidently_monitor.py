"""Monitor de covariate shift del sistema CARDIS (Hito 5 — Bloque 4).

Servicio FastAPI de proceso único que, cada ``CARDIS_DRIFT_WINDOW_MIN``
minutos, descarga el dataset de referencia y las inferencias recientes
desde MinIO, ejecuta ``Evidently DataDriftPreset`` y actualiza tres
métricas Prometheus:

* ``cardis_drift_dataset``               — 1 si hay drift a nivel dataset.
* ``cardis_drift_share_columns``         — fracción de features con drift.
* ``cardis_drift_column{feature=<name>}``— 1/0 por feature individual.
* ``cardis_drift_last_run_timestamp_seconds`` — timestamp del último ciclo.

Variables de entorno relevantes
---------------------------------
MINIO_ENDPOINT          host:puerto del servidor MinIO.
MINIO_ACCESS_KEY        clave de acceso MinIO.
MINIO_SECRET_KEY        clave secreta MinIO.
MINIO_SECURE            "true" para HTTPS (default: false).
MINIO_BUCKET            bucket de modelos para cargar FeatureBuilder
                        (default: cardis-models).
CARDIS_INFERENCES_BUCKET bucket con inferencias (default: cardis-inferences).
CARDIS_DATA_BUCKET       bucket con reference.parquet (default: cardis-data).
CARDIS_DRIFT_WINDOW_MIN  ventana en minutos (default: 5).

Notas de implementación
------------------------
* **Orden de imports**: prometheus_client SE IMPORTA ANTES que config.py.
  config.py tiene un side-effect que setea PROMETHEUS_MULTIPROC_DIR.
  Si prometheus_client ya está inicializado antes de ese seteo, queda en
  modo single-process — que es exactamente lo que queremos aquí (proceso
  único, sin workers). Relación inversa a src/api.py (ver hito5_notes.md).
* **Lectura de parquets**: cada fichero se lee con ``pq.ParquetFile(path)``
  individualmente para evitar que PyArrow infiera la columna virtual ``dt``
  del directorio Hive-particionado (ver nota B2 en hito5_notes.md).
* **Boundary de medianoche**: si la hora actual < ventana, se listan también
  los objetos del día anterior para no perder registros recientes.
* **Resiliencia**: el bucle ``drift_loop`` captura cualquier excepción,
  loguea y continúa. ``_LAST_RUN`` se actualiza en ``finally`` siempre.
* **Primer ciclo inmediato**: sin sleep previo al arrancar.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# ATENCIÓN: prometheus_client DEBE importarse ANTES de cualquier módulo local.
# config.py tiene un side-effect de módulo que setea PROMETHEUS_MULTIPROC_DIR;
# si prometheus_client ya está inicializado en ese momento, lo ignora y
# queda en modo single-process (correcto para este servicio de 1 worker).
# ─────────────────────────────────────────────────────────────────────────────
import prometheus_client  # noqa: F401  — inicialización single-process

from prometheus_client import REGISTRY, Gauge, generate_latest  # noqa: E402

# ─── Stdlib ──────────────────────────────────────────────────────────────────
import asyncio
import os
import tempfile
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# ─── Terceros ────────────────────────────────────────────────────────────────
import pandas as pd
import pyarrow.parquet as pq
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from minio import Minio

# ─── Locales (después de prometheus_client) ──────────────────────────────────
from .config import StorageConfig, setup_logging
from .features import FeatureBuilder

logger = setup_logging()

# ─────────────────────────────────────────────────────────────────────────────
# Métricas Prometheus (single-process — sin multiprocess_mode)
# ─────────────────────────────────────────────────────────────────────────────
_DRIFT_DATASET = Gauge(
    "cardis_drift_dataset",
    "1 si Evidently detecta dataset-level drift, 0 si no",
)
_DRIFT_SHARE = Gauge(
    "cardis_drift_share_columns",
    "Fracción de features con drift detectado (0-1)",
)
_DRIFT_COLUMN = Gauge(
    "cardis_drift_column",
    "1 si la feature tiene drift, 0 si no",
    ["feature"],
)
_LAST_RUN = Gauge(
    "cardis_drift_last_run_timestamp_seconds",
    "Timestamp Unix (epoch segundos) de la última ejecución del ciclo de deriva",
)

# ─────────────────────────────────────────────────────────────────────────────
# Estado global
# ─────────────────────────────────────────────────────────────────────────────
_FEATURE_COLS: list[str] = []
_ready: bool = False

MIN_SAMPLES: int = 5  # mínimo de registros en ventana para ejecutar Evidently


# ─────────────────────────────────────────────────────────────────────────────
# Helpers MinIO
# ─────────────────────────────────────────────────────────────────────────────

def _build_minio_client() -> Minio:
    """Construye el cliente MinIO con la configuración del entorno."""
    sc = StorageConfig()
    return Minio(
        sc.endpoint,
        access_key=sc.access_key,
        secret_key=sc.secret_key,
        secure=sc.secure,
    )


def _load_feature_builder(client: Minio) -> FeatureBuilder:
    """Descarga el FeatureBuilder desde MinIO y lo deserializa.

    Parámetros
    ----------
    client : Minio
        Cliente MinIO ya configurado.

    Devuelve
    --------
    FeatureBuilder
        Instancia con ``feature_names_`` disponible.
    """
    models_bucket = os.getenv("MINIO_BUCKET", "cardis-models")
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        client.fget_object(models_bucket, "cardis_featurebuilder.joblib", str(tmp_path))
        return FeatureBuilder.load(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _download_reference(
    client: Minio,
    data_bucket: str,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Descarga ``reference.parquet`` desde MinIO.

    Parámetros
    ----------
    client : Minio
    data_bucket : str
    feature_cols : list[str]
        Whitelist de columnas a conservar.

    Devuelve
    --------
    pd.DataFrame
        Solo las columnas ``feature_cols``.
    """
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        client.fget_object(data_bucket, "reference.parquet", str(tmp_path))
        # pq.ParquetFile evita la columna virtual 'dt' de Hive-partitioning
        df = pq.ParquetFile(str(tmp_path)).read().to_pandas()
    finally:
        tmp_path.unlink(missing_ok=True)

    available = [c for c in feature_cols if c in df.columns]
    return df[available].copy()


def _download_current_window(
    client: Minio,
    infer_bucket: str,
    window_min: int,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Descarga inferencias de la ventana temporal actual.

    Lista objetos de HOY (y AYER si la hora < ventana) para manejar el
    boundary de medianoche. Cada fichero se lee individualmente para evitar
    la columna virtual ``dt`` de PyArrow ≥ 14 (nota B2, hito5_notes.md).

    Parámetros
    ----------
    client : Minio
    infer_bucket : str
    window_min : int
        Tamaño de la ventana en minutos.
    feature_cols : list[str]
        Whitelist de columnas a conservar.

    Devuelve
    --------
    pd.DataFrame
        Filas dentro de la ventana, solo columnas ``feature_cols``.
        DataFrame vacío (con columnas) si no hay datos.
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(minutes=window_min)

    # Prefijos a listar: siempre HOY, AYER si estamos en la primera ventana
    prefixes: list[str] = [f"dt={now.strftime('%Y-%m-%d')}/"]
    if now.hour == 0 and now.minute < window_min:
        yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")
        prefixes.append(f"dt={yesterday}/")

    frames: list[pd.DataFrame] = []
    for prefix in prefixes:
        try:
            objects = list(client.list_objects(infer_bucket, prefix=prefix))
        except Exception as exc:  # noqa: BLE001
            logger.warning("No se pudieron listar objetos con prefijo %s: %s", prefix, exc)
            continue

        for obj in objects:
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                client.fget_object(infer_bucket, obj.object_name, str(tmp_path))
                # Lee sin prefijo de directorio → no columna virtual 'dt'
                df_part = pq.ParquetFile(str(tmp_path)).read().to_pandas()
                frames.append(df_part)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error al descargar %s: %s", obj.object_name, exc)
            finally:
                tmp_path.unlink(missing_ok=True)

    if not frames:
        return pd.DataFrame(columns=feature_cols)

    combined = pd.concat(frames, ignore_index=True)

    # Filtrar por ventana temporal usando la columna timestamp del parquet
    if "timestamp" in combined.columns:
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
        combined = combined[combined["timestamp"] >= cutoff]

    available = [c for c in feature_cols if c in combined.columns]
    return combined[available].copy() if available else pd.DataFrame(columns=feature_cols)


# ─────────────────────────────────────────────────────────────────────────────
# Lógica Evidently
# ─────────────────────────────────────────────────────────────────────────────

def _run_evidently(
    ref: pd.DataFrame,
    cur: pd.DataFrame,
) -> dict[str, Any]:
    """Ejecuta DataDriftPreset y extrae los resultados relevantes.

    Parámetros
    ----------
    ref : pd.DataFrame
        Dataset de referencia (entrenamiento post-featurización).
    cur : pd.DataFrame
        Dataset actual (ventana de inferencias).

    Devuelve
    --------
    dict con claves:
        dataset_drift : bool
        share : float   (0-1)
        columns : dict{nombre: bool}
    """
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)
    raw = report.as_dict()
    metrics = raw["metrics"]

    # DatasetDriftMetric → dataset_drift + share_of_drifted_columns
    dataset_m = next(
        m for m in metrics if m["metric"] == "DatasetDriftMetric"
    )
    r = dataset_m["result"]

    # DataDriftTable → drift_by_columns (drift detallado por feature)
    table_m = next(m for m in metrics if m["metric"] == "DataDriftTable")
    drift_by_col: dict[str, dict[str, Any]] = table_m["result"]["drift_by_columns"]

    return {
        "dataset_drift": bool(r["dataset_drift"]),
        "share": float(r["share_of_drifted_columns"]),
        "columns": {
            col: bool(info["drift_detected"])
            for col, info in drift_by_col.items()
        },
    }


def _update_gauges(result: dict[str, Any]) -> None:
    """Actualiza los tres Gauges de deriva con el resultado de Evidently."""
    _DRIFT_DATASET.set(1.0 if result["dataset_drift"] else 0.0)
    _DRIFT_SHARE.set(result["share"])
    for col, drifted in result["columns"].items():
        _DRIFT_COLUMN.labels(feature=col).set(1.0 if drifted else 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Ciclo principal de deriva
# ─────────────────────────────────────────────────────────────────────────────

async def drift_loop() -> None:
    """Ciclo asyncio que ejecuta la detección de deriva cada ``window_min`` min.

    Diseño:
    - Primer ciclo inmediato al arrancar (sin sleep inicial).
    - Cuerpo en try/except para que ninguna excepción mate el bucle.
    - ``_LAST_RUN`` se actualiza SIEMPRE en ``finally`` (éxito o fallo).
    - ``_ready`` se pone a True tras el primer ciclo completado.
    """
    global _ready

    data_bucket = os.getenv("CARDIS_DATA_BUCKET", "cardis-data")
    infer_bucket = os.getenv("CARDIS_INFERENCES_BUCKET", "cardis-inferences")
    window_min = int(os.getenv("CARDIS_DRIFT_WINDOW_MIN", "5"))

    logger.info(
        "drift_loop arranca: bucket_data=%s bucket_inferences=%s ventana=%d min",
        data_bucket, infer_bucket, window_min,
    )

    while True:
        try:
            client = _build_minio_client()
            ref = _download_reference(client, data_bucket, _FEATURE_COLS)
            cur = _download_current_window(client, infer_bucket, window_min, _FEATURE_COLS)

            if len(cur) < MIN_SAMPLES:
                logger.warning(
                    "Ventana actual contiene %d registros (mínimo %d); ciclo omitido",
                    len(cur), MIN_SAMPLES,
                )
            else:
                result = _run_evidently(ref, cur)
                _update_gauges(result)
                logger.info(
                    "Deriva calculada: dataset_drift=%s share=%.3f "
                    "features_con_drift=%d/%d",
                    result["dataset_drift"],
                    result["share"],
                    sum(result["columns"].values()),
                    len(result["columns"]),
                )

            _ready = True

        except Exception:  # noqa: BLE001
            logger.exception("Error en ciclo de deriva — continuando en el siguiente ciclo")

        finally:
            _LAST_RUN.set(time.time())

        await asyncio.sleep(window_min * 60)


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

# Fallback hardcoded: 36 features conocidas del FeatureBuilder (Hito 2/3)
_HARDCODED_FEATURE_COLS: list[str] = [
    "edad", "altura_cm", "peso_kg", "imc",
    "presion_sistolica_1", "presion_sistolica_2", "presion_sistolica_3",
    "colesterol_total", "ldl", "hdl", "glucosa_ayunas",
    "fumador", "antecedentes_familiares",
    "glucosa_missing", "notas_missing", "actividad_missing",
    "hospital_te", "horas_actividad",
    "anio_visita", "mes_visita", "dia_semana", "hora_visita", "es_fin_semana",
    "kw_diabetes", "kw_infarto", "kw_coronario", "kw_mareo",
    "kw_pecho", "kw_fatiga", "kw_dolor", "kw_sedentarismo",
    "kw_angina", "kw_fumador",
    "presion_sistolica_media", "presion_sistolica_mediana", "presion_sistolica_std",
]


@asynccontextmanager
async def lifespan(application: FastAPI):  # noqa: ARG001
    """Carga el FeatureBuilder y arranca el ciclo de deriva."""
    global _FEATURE_COLS

    try:
        client = _build_minio_client()
        fb = _load_feature_builder(client)
        _FEATURE_COLS = list(fb.feature_names_)
        logger.info(
            "FeatureBuilder cargado desde MinIO: %d features", len(_FEATURE_COLS)
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "No se pudo cargar FeatureBuilder (%s); usando lista hardcoded de %d features",
            exc, len(_HARDCODED_FEATURE_COLS),
        )
        _FEATURE_COLS = list(_HARDCODED_FEATURE_COLS)

    asyncio.create_task(drift_loop())
    yield


app = FastAPI(
    title="CARDIS Drift Monitor",
    description=(
        "Servicio de detección de covariate shift con Evidently "
        "(Hito 5 — Bloque 4). Expone métricas Prometheus en /metrics."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/v2/health/live")
async def health_live() -> dict[str, str]:
    """Liveness probe — siempre 200 si el proceso está vivo."""
    return {"status": "live"}


@app.get("/v2/health/ready")
async def health_ready() -> dict[str, str]:
    """Readiness probe — 503 hasta completar el primer ciclo de deriva."""
    if not _ready:
        raise HTTPException(
            status_code=503,
            detail="Primer ciclo de deriva pendiente",
        )
    return {"status": "ready"}


@app.get("/metrics")
async def metrics_endpoint() -> Response:
    """Endpoint de métricas Prometheus (single-process)."""
    return Response(
        content=generate_latest(REGISTRY),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
