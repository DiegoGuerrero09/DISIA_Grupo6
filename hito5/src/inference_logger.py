"""Persistencia asíncrona de inferencias CARDIS en MinIO (Hito 5 — Bloque 2).

Cada llamada a /predict escribe un InferenceRecord en el buffer en memoria.
Un bucle de fondo lo descarga a MinIO en Parquet cuando se cumple alguna
de las condiciones de flush (por tamaño, por tiempo o por espera máxima).

El InferenceLogger nunca interrumpe /predict: cualquier error de MinIO
se loguea y se descarta silenciosamente.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, TypedDict
from uuid import uuid4

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from minio import Minio

from .config import setup_logging

logger = setup_logging()


class InferenceRecord(TypedDict):
    """Registro tipado de una inferencia individual (un paciente).

    Contiene 5 campos de metadatos, 36 features post-featurización y
    3 campos de resultado de la predicción (40 campos en total).
    """

    # Metadatos de la petición
    timestamp: str         # ISO-8601 UTC del momento de la petición
    model_version: str
    request_id: str        # UUID4 de la petición HTTP
    worker_pid: int
    patient_index: int     # 0-based dentro de la petición

    # Features post-featurización (FeatureBuilder.transform)
    edad: float
    altura_cm: float
    peso_kg: float
    imc: float
    presion_sistolica_1: float
    presion_sistolica_2: float
    presion_sistolica_3: float
    colesterol_total: float
    ldl: float
    hdl: float
    glucosa_ayunas: float
    fumador: float
    antecedentes_familiares: float
    glucosa_missing: float
    notas_missing: float
    actividad_missing: float
    hospital_te: float
    horas_actividad: float
    anio_visita: float
    mes_visita: float
    dia_semana: float
    hora_visita: float
    es_fin_semana: float
    kw_diabetes: float
    kw_infarto: float
    kw_coronario: float
    kw_mareo: float
    kw_pecho: float
    kw_fatiga: float
    kw_dolor: float
    kw_sedentarismo: float
    kw_angina: float
    kw_fumador: float
    presion_sistolica_media: float
    presion_sistolica_mediana: float
    presion_sistolica_std: float

    # Resultado de la predicción
    probability: float
    label: int
    age_warning: bool


class InferenceLogger:
    """Buffer en memoria con flush periódico a MinIO en formato Parquet.

    Parámetros
    ----------
    bucket : str
        Nombre del bucket MinIO destino.
    endpoint : str
        Dirección del servidor MinIO (``host:puerto``, sin esquema).
    access_key : str
        Clave de acceso MinIO.
    secret_key : str
        Clave secreta MinIO.
    secure : bool
        Si True usa HTTPS, si False usa HTTP. Por defecto False.
    flush_size : int
        Número de registros que activa el flush por tamaño. Defecto: 50.
    flush_interval_s : float
        Segundos de espera con ``>= min_flush_size`` registros. Defecto: 60.
    min_flush_size : int
        Mínimo de registros para flush por tiempo. Defecto: 5.
    max_flush_wait_s : float
        Segundos máximos de espera con cualquier registro. Defecto: 300.
    """

    def __init__(
        self,
        bucket: str,
        endpoint: str,
        access_key: str,
        secret_key: str,
        secure: bool = False,
        flush_size: int = 50,
        flush_interval_s: float = 60.0,
        min_flush_size: int = 5,
        max_flush_wait_s: float = 300.0,
        _check_interval_s: float = 10.0,
    ) -> None:
        self._bucket = bucket
        self._endpoint = endpoint
        self._access_key = access_key
        self._secret_key = secret_key
        self._secure = secure
        self._flush_size = flush_size
        self._flush_interval_s = flush_interval_s
        self._min_flush_size = min_flush_size
        self._max_flush_wait_s = max_flush_wait_s
        self._check_interval_s = _check_interval_s

        self._buffer: List[InferenceRecord] = []
        self._first_record_ts: Optional[float] = None  # time.monotonic()
        self._lock: asyncio.Lock = asyncio.Lock()
        self._task: Optional[asyncio.Task] = None  # type: ignore[type-arg]

    # ------------------------------------------------------------------
    # API pública (síncrona)
    # ------------------------------------------------------------------

    def log(self, record: InferenceRecord) -> None:
        """Añade un registro al buffer (síncrono, sin IO)."""
        if self._first_record_ts is None:
            self._first_record_ts = time.monotonic()
        self._buffer.append(record)

    def _should_flush(self, n: int, elapsed: float) -> bool:
        """Devuelve True si alguna condición de flush se cumple.

        Parámetros
        ----------
        n : int
            Número de registros en el buffer.
        elapsed : float
            Segundos desde que se añadió el primer registro del buffer.
        """
        return (
            n >= self._flush_size
            or (n >= self._min_flush_size and elapsed >= self._flush_interval_s)
            or (n >= 1 and elapsed >= self._max_flush_wait_s)
        )

    # ------------------------------------------------------------------
    # API asíncrona
    # ------------------------------------------------------------------

    async def _do_flush(self) -> None:
        """Vacía el buffer y sube los registros a MinIO como Parquet."""
        async with self._lock:
            if not self._buffer:
                return
            records = list(self._buffer)
            self._buffer.clear()
            self._first_record_ts = None

        # IO completamente fuera del lock
        n = len(records)
        tmp_path: Optional[Path] = None
        try:
            df = pd.DataFrame(records)
            table = pa.Table.from_pandas(df, preserve_index=False)

            ts = datetime.now(timezone.utc)
            date_str = ts.strftime("%Y-%m-%d")
            time_str = ts.strftime("%H%M%S")
            uuid_short = str(uuid4())[:8]
            pid = os.getpid()
            object_name = f"dt={date_str}/{time_str}-{pid}-{uuid_short}.parquet"

            tmp_fd, tmp_name = tempfile.mkstemp(suffix=".parquet")
            os.close(tmp_fd)
            tmp_path = Path(tmp_name)

            pq.write_table(table, tmp_path, compression="snappy")

            client = Minio(
                self._endpoint,
                access_key=self._access_key,
                secret_key=self._secret_key,
                secure=self._secure,
            )
            client.fput_object(self._bucket, object_name, str(tmp_path))
            logger.info(
                "InferenceLogger: subidos %d registros → %s", n, object_name
            )
        except Exception:
            logger.exception(
                "InferenceLogger: flush fallido, descartando %d registros", n
            )
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()

    async def flush(self) -> None:
        """Flush público; vacía el buffer aunque esté por debajo del mínimo."""
        await self._do_flush()

    async def _background_loop(self) -> None:
        """Bucle de fondo que evalúa las condiciones de flush periódicamente."""
        while True:
            await asyncio.sleep(self._check_interval_s)
            n = len(self._buffer)
            if n == 0:
                continue
            first_ts = self._first_record_ts
            if first_ts is None:
                continue
            elapsed = time.monotonic() - first_ts
            if self._should_flush(n, elapsed):
                await self._do_flush()

    async def start_background_task(self) -> None:
        """Inicia el bucle de fondo. Llamar desde el lifespan de FastAPI."""
        self._task = asyncio.create_task(self._background_loop())
        logger.info("InferenceLogger: background task iniciado")

    async def stop(self) -> None:
        """Cancela el bucle de fondo y hace un flush final del buffer."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self._do_flush()
        logger.info("InferenceLogger: detenido, buffer vaciado")
