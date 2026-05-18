from flask import Flask, request
from kubernetes import client, config
import os

app = Flask(__name__)

@app.route('/retrain', methods=['POST'])
def retrain():
    data = request.json
    print(f" Alerta recibida: {data.get('status')}")

    if data.get('status') == 'firing':
        config.load_incluster_config()
        _ = client.BatchV1Api()
        
        job_name = os.getenv("JOB_NAME", "cardis-trainer")
        namespace = os.getenv("NAMESPACE", "cardis")
        
        print(f"Disparando reentreno: {job_name} en {namespace}...")
        
    return {"message": "Reentreno iniciado"}, 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)