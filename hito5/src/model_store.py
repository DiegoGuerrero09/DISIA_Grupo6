"""Capa de almacenamiento de artefactos (modelo + featurizer + metadatos).

Soporta dos backends:

* ``local``: el artefacto se persiste en disco (montaje compartido o
  volumen de Docker). Ideal para desarrollo y para los Jobs de Kubernetes.
* ``minio``: el artefacto se sube a un bucket S3-compatible (MinIO). Es
  el mecanismo recomendado en produccion porque desacopla los
  contenedores de entrenamiento e inferencia.

El backend se selecciona mediante la variable ``CARDIS_STORAGE_BACKEND``.
"""

from __future__ import annotations

import fcntl
import io
import shutil
from pathlib import Path
from typing import Iterable

from .config import StorageConfig, setup_logging

logger = setup_logging()


class ModelStore:
    """Interfaz unificada para subir/descargar artefactos."""

    def __init__(self, config: StorageConfig | None = None) -> None:
        self.config = config or StorageConfig()
        self._client = None
        if self.config.backend == "minio":
            self._client = self._build_minio_client()

    # ---------------------------------------------------------------- minio
    def _build_minio_client(self):
        try:
            from minio import Minio  # type: ignore
        except ImportError as exc:  # pragma: no cover - depende del entorno
            raise RuntimeError(
                "Para usar el backend MinIO instala la dependencia 'minio'."
            ) from exc

        client = Minio(
            self.config.endpoint,
            access_key=self.config.access_key,
            secret_key=self.config.secret_key,
            secure=self.config.secure,
        )
        if not client.bucket_exists(self.config.bucket):
            client.make_bucket(self.config.bucket)
            logger.info("Bucket MinIO '%s' creado", self.config.bucket)
        return client

    # ---------------------------------------------------------------- API
    def put(self, local_path: Path, remote_name: str | None = None) -> str:
        """Persiste ``local_path`` en el backend y devuelve el URI logico."""
        local_path = Path(local_path)
        remote_name = remote_name or local_path.name
        if self.config.backend == "minio":
            assert self._client is not None
            self._client.fput_object(
                self.config.bucket, remote_name, str(local_path)
            )
            uri = f"s3://{self.config.bucket}/{remote_name}"
            logger.info("Artefacto subido a %s", uri)
            return uri
        # local
        target = local_path  # ya esta en disco
        logger.info("Artefacto disponible localmente en %s", target)
        return str(target)

    def get(self, remote_name: str, local_path: Path) -> Path:
        """Descarga ``remote_name`` al ``local_path`` indicado.

        Usa un fichero lock por proceso para evitar la race condition cuando
        uvicorn arranca con ``--workers > 1`` y varios workers intentan
        descargar el mismo artefacto simultáneamente.
        """
        # Resolver a ruta absoluta para que fget_object (minio-py) coloque el
        # fichero temporal en el directorio correcto independientemente del CWD
        # del proceso (importante con --workers > 1 en uvicorn).
        local_path = Path(local_path).resolve()
        if local_path.exists():
            logger.info("Artefacto %s ya existe en disco, se omite descarga", remote_name)
            return local_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if self.config.backend == "minio":
            assert self._client is not None
            lock_path = local_path.with_suffix(local_path.suffix + ".lock")
            with lock_path.open("w") as _lock:
                fcntl.flock(_lock, fcntl.LOCK_EX)
                if not local_path.exists():
                    self._client.fget_object(
                        self.config.bucket, remote_name, str(local_path)
                    )
                    logger.info(
                        "Descargado %s desde MinIO a %s", remote_name, local_path
                    )
                else:
                    logger.info(
                        "Artefacto %s descargado por otro worker, reutilizando", remote_name
                    )
            return local_path
        # local: si remote_name == local_path, no hay nada que hacer
        candidate = Path(remote_name)
        if candidate.exists() and candidate.resolve() != local_path.resolve():
            shutil.copy2(candidate, local_path)
        return local_path

    def put_many(self, paths: Iterable[Path]) -> list[str]:
        return [self.put(p) for p in paths]
