"""Genera el dataset de referencia para la detección de deriva (Hito 5 — B4).

Carga el CSV de entrenamiento, aplica el ``FeatureBuilder`` ya entrenado
y persiste el resultado como ``reference.parquet`` en el bucket
``cardis-data`` de MinIO.

Uso típico (desde el directorio ``hito5/``)::

    python scripts/generate_reference_dataset.py

    # Con endpoint personalizado (port-forward al cluster):
    python scripts/generate_reference_dataset.py \\
        --endpoint localhost:9100 \\
        --bucket cardis-data

El script es idempotente: si ``reference.parquet`` ya existe se sobreescribe.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
from minio import Minio
from minio.error import S3Error


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Genera reference.parquet y lo sube a MinIO.",
    )
    parser.add_argument(
        "--csv",
        default=str(Path(__file__).parent.parent.parent / "data" / "raw" / "cardio_risk_train.csv"),
        help="Ruta al CSV de entrenamiento (default: ../../../data/raw/cardio_risk_train.csv).",
    )
    parser.add_argument(
        "--feature-builder",
        default="models/cardis_featurebuilder.joblib",
        help="Ruta al artefacto FeatureBuilder serializado (default: models/cardis_featurebuilder.joblib).",
    )
    parser.add_argument(
        "--endpoint",
        default=os.getenv("MINIO_ENDPOINT", "localhost:9000"),
        help="Endpoint MinIO host:puerto (default: MINIO_ENDPOINT env o localhost:9000).",
    )
    parser.add_argument(
        "--bucket",
        default=os.getenv("CARDIS_DATA_BUCKET", "cardis-data"),
        help="Bucket destino (default: CARDIS_DATA_BUCKET env o cardis-data).",
    )
    parser.add_argument(
        "--access",
        default=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        help="MinIO access key.",
    )
    parser.add_argument(
        "--secret",
        default=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        help="MinIO secret key.",
    )
    return parser


def generate(
    csv_path: Path,
    fb_path: Path,
    endpoint: str,
    bucket: str,
    access: str,
    secret: str,
) -> None:
    """Genera ``reference.parquet`` y lo sube a MinIO.

    Parámetros
    ----------
    csv_path : Path
        Ruta al CSV de entrenamiento (datos crudos).
    fb_path : Path
        Ruta al FeatureBuilder serializado (.joblib).
    endpoint : str
        Dirección del servidor MinIO (``host:puerto``).
    bucket : str
        Nombre del bucket destino.
    access, secret : str
        Credenciales MinIO.
    """
    # ── 1. Cargar CSV ───────────────────────────────────────────────────────
    if not csv_path.exists():
        print(f"[ERROR] CSV no encontrado: {csv_path}", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] Cargando CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[INFO] Filas cargadas: {len(df)}")

    # ── 2. Cargar FeatureBuilder ────────────────────────────────────────────
    # Añadir el directorio padre (hito5/) al path para poder importar src.*
    hito5_dir = Path(__file__).parent.parent
    if str(hito5_dir) not in sys.path:
        sys.path.insert(0, str(hito5_dir))

    from src.features import FeatureBuilder  # noqa: PLC0415

    if not fb_path.exists():
        print(f"[ERROR] FeatureBuilder no encontrado: {fb_path}", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] Cargando FeatureBuilder: {fb_path}")
    fb = FeatureBuilder.load(fb_path)
    print(f"[INFO] Features del modelo: {len(fb.feature_names_)}")

    # ── 3. Transformar dataset ──────────────────────────────────────────────
    print("[INFO] Aplicando FeatureBuilder.transform()...")
    reference_df: pd.DataFrame = fb.transform(df)
    print(f"[INFO] Referencia generada: {reference_df.shape}")

    # ── 4. Subir a MinIO ────────────────────────────────────────────────────
    client = Minio(endpoint, access_key=access, secret_key=secret, secure=False)

    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        print(f"[INFO] Bucket '{bucket}' creado")
    else:
        print(f"[INFO] Bucket '{bucket}' ya existe")

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        reference_df.to_parquet(tmp_path, index=False)
        size_kb = tmp_path.stat().st_size // 1024
        print(f"[INFO] Parquet temporal: {tmp_path} ({size_kb} KB)")

        client.fput_object(bucket, "reference.parquet", str(tmp_path))
        print(f"[INFO] Subido: s3://{bucket}/reference.parquet ({size_kb} KB)")
    except S3Error as exc:
        print(f"[ERROR] Error de MinIO: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        tmp_path.unlink(missing_ok=True)

    print("[OK] reference.parquet listo en MinIO.")


def main() -> None:
    args = _build_parser().parse_args()
    generate(
        csv_path=Path(args.csv),
        fb_path=Path(args.feature_builder),
        endpoint=args.endpoint,
        bucket=args.bucket,
        access=args.access,
        secret=args.secret,
    )


if __name__ == "__main__":
    main()
