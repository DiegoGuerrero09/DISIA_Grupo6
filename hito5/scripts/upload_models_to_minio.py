"""Sube los artefactos del modelo a una instancia MinIO.

Uso principal: poblar la MinIO de Kubernetes con los modelos entrenados
localmente, para que el Deployment del inferer pueda arrancar.

    python scripts/upload_models_to_minio.py \
        --models-dir models/ \
        --endpoint localhost:9100 \
        --bucket cardis-models

El puerto 9100 es el local cuando se hace port-forward al MinIO de K8s:
    kubectl port-forward -n cardis statefulsets/minio 9100:9000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from minio import Minio
from minio.error import S3Error


def upload(models_dir: Path, endpoint: str, bucket: str, access: str, secret: str) -> None:
    client = Minio(endpoint, access_key=access, secret_key=secret, secure=False)
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        print(f"Bucket '{bucket}' creado")

    artifacts = list(models_dir.glob("*.joblib")) + list(models_dir.glob("*.json"))
    if not artifacts:
        print(f"No se encontraron artefactos en {models_dir}")
        return

    for path in artifacts:
        client.fput_object(bucket, path.name, str(path))
        print(f"Subido: {path.name}  ({path.stat().st_size // 1024} KB)")

    print(f"\nTotal: {len(artifacts)} artefactos en s3://{bucket}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sube modelos a MinIO")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--endpoint", default="localhost:9100")
    parser.add_argument("--bucket", default="cardis-models")
    parser.add_argument("--access-key", default="minioadmin")
    parser.add_argument("--secret-key", default="minioadmin")
    args = parser.parse_args()

    upload(args.models_dir, args.endpoint, args.bucket, args.access_key, args.secret_key)


if __name__ == "__main__":
    main()
