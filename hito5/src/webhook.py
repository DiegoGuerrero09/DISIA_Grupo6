import io

from flask import Flask, request
from kubernetes import client, config
import json
from minio import Minio
from minio.error import S3Error
import time
import os

def get_minio_client():
    return Minio(
        os.getenv("MINIO_ENDPOINT", "minio:9000"),
        access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        secure=False
    )

def ensure_bucket(client, bucket_name: str):
    found = client.bucket_exists(bucket_name)
    if not found:
        client.make_bucket(bucket_name)

app = Flask(__name__)

def create_training_job():
    config.load_incluster_config()
    batch = client.BatchV1Api()

    namespace = os.getenv("NAMESPACE", "cardis")

    job_name = f"cardis-train-{int(time.time())}"

    job = client.V1Job(
        metadata=client.V1ObjectMeta(name=job_name),
        spec=client.V1JobSpec(
            template=client.V1PodTemplateSpec(
                spec=client.V1PodSpec(
                    restart_policy="Never",
                    containers=[
                        client.V1Container(
                            name="trainer",
                            image=os.getenv("TRAIN_IMAGE", "cardis/train:1.0.0"),
                            env=[
                                client.V1EnvVar(
                                    name="LOAD_FROM_MINIO",
                                    value="true"
                                )
                            ]
                        )
                    ]
                )
            )
        )
    )

    batch.create_namespaced_job(namespace=namespace, body=job)

@app.route("/alert", methods=["POST"])
def alert():
    data = request.json

    if data.get("status") == "firing":
        try:
            create_training_job()
            return {"status": "training triggered"}, 200
        except Exception as e:
            return {"error": str(e)}, 500

    return {"status": "ignored"}, 200

@app.route("/groundtruth", methods=["POST"])
def groundtruth():
    data = request.json

    if not data:
        return {"error": "empty payload"}, 400

    bucket = os.getenv("CARDIS_GROUNDTRUTH_BUCKET", "cardis-groundtruth")

    try:
        client = get_minio_client()
        ensure_bucket(client, bucket)

        object_name = f"groundtruth-{int(time.time())}.json"

        payload = json.dumps(data).encode("utf-8")

        client.put_object(
            bucket,
            object_name,
            data=io.BytesIO(payload),
            length=len(payload),
            content_type="application/json"
        )

        print(f"Groundtruth guardado en MinIO: {object_name}")

        return {"status": "stored", "object": object_name}, 200

    except S3Error as e:
        print("MinIO error:", e)
        return {"error": str(e)}, 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)