import io
import json
import os

import pandas as pd
from minio import Minio

ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9001")
ACCESS = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
SECRET = os.getenv("MINIO_SECRET_KEY", "minioadmin")

client = Minio(ENDPOINT, access_key=ACCESS, secret_key=SECRET, secure=False)

def load_inferences(bucket):
    """Carga parquets de inferencia particionados."""
    objects = client.list_objects(bucket, recursive=True)
    data = []
    for obj in objects:
        if obj.object_name.endswith(".parquet"):
            res = client.get_object(bucket, obj.object_name)
            data.append(pd.read_parquet(io.BytesIO(res.read())))
    return pd.concat(data, ignore_index=True) if data else pd.DataFrame()

def load_groundtruth(bucket):
    """Carga JSONs de etiquetas reales."""
    objects = client.list_objects(bucket)
    rows = []
    for obj in objects:
        if obj.object_name.endswith(".json"):
            res = client.get_object(bucket, obj.object_name)
            item = json.loads(res.read())
            rows.append(item)
    return pd.DataFrame(rows)

print("Cargando datos desde MinIO...")
preds = load_inferences("cardis-inferences")
truth = load_groundtruth("cardis-groundtruth")

if preds.empty or truth.empty:
    print("No hay datos suficientes para el join.")
else:
    df = preds.merge(truth, on="request_id", suffixes=("_pred", "_true"))

    if "label_true" in df.columns:
        df["correct"] = df["label_pred"] == df["label_true"]
        accuracy = df["correct"].mean()
        print(f"Accuracy actual del sistema: {accuracy:.2%}")
        print(f"Total registros pareados: {len(df)}")
    else:
        print("Error: No se encontró la columna 'label_true' en el groundtruth.")
        print("Columnas disponibles:", df.columns.tolist())
