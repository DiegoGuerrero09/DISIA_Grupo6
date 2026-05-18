"""Modulo de ingestion de datos del sistema CARDIS.

Responsable de cargar el dataset clinico desde la fuente externa (en este
caso un CSV, pero en un escenario real podria ser una base de datos
hospitalaria o el HIS) y exponerlo al pipeline mediante una funcion
unica. Mantiene la separacion entre la lectura del dato crudo y su
transformacion posterior, lo que facilita reemplazar la fuente sin tocar
el resto del pipeline.

Uso por linea de comandos::

    python -m src.data_ingestion --input data/raw/cardis_train.csv \
        --output data/processed/cardis_raw.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import Paths, setup_logging

logger = setup_logging()


def load_csv(path: Path) -> pd.DataFrame:
    """Carga un CSV de pacientes desde disco.

    Parameters
    ----------
    path : Path
        Ruta al fichero CSV de entrada.

    Returns
    -------
    pandas.DataFrame
        Dataframe con los registros tal cual se reciben de la fuente.
    """
    if not path.exists():
        raise FileNotFoundError(f"No se encuentra el dataset en: {path}")
    logger.info("Cargando dataset desde %s", path)
    df = pd.read_csv(path)
    logger.info("Dataset cargado: %d filas, %d columnas", *df.shape)
    return df


def save_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Persiste el dataframe en formato parquet (mas eficiente que csv)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Dataset persistido en %s", path)
    return path


def ingest(input_path: Path, output_path: Path) -> Path:
    """Pipeline minimo de ingestion: leer y persistir en parquet."""
    df = load_csv(input_path)
    return save_parquet(df, output_path)


def load_from_minio(bucket: str, key: str) -> pd.DataFrame:
    """Descarga un CSV desde MinIO y lo devuelve como DataFrame.

    Parámetros
    ----------
    bucket : str
        Nombre del bucket MinIO.
    key : str
        Clave (ruta) del objeto dentro del bucket.

    Notas
    -----
    Función preparada para el Bloque 6 (Decisión 4 del Hito 5).
    No se invoca desde ningún módulo todavía.
    """
    import io

    from minio import Minio

    from .config import StorageConfig

    sc = StorageConfig()
    client = Minio(
        sc.endpoint,
        access_key=sc.access_key,
        secret_key=sc.secret_key,
        secure=sc.secure,
    )
    obj = client.get_object(bucket, key)
    try:
        return pd.read_csv(io.BytesIO(obj.read()))
    finally:
        obj.close()
        obj.release_conn()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingestion de datos del sistema CARDIS"
    )
    paths = Paths()
    parser.add_argument(
        "--input",
        type=Path,
        default=paths.raw_train,
        help="CSV de entrada (por defecto: variable CARDIS_RAW_TRAIN)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=paths.processed_dir / "cardis_raw.parquet",
        help="Parquet de salida (por defecto: ${CARDIS_PROCESSED_DIR}/cardis_raw.parquet)",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    ingest(args.input, args.output)


if __name__ == "__main__":
    main()
