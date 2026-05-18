#!/usr/bin/env bash
# Test end-to-end basico: arranca el stack, entrena, valida health-check
# y manda una peticion sintetica al endpoint /predict. Util para CI/CD
# y para verificar el despliegue tras un rebuild.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "[1/5] Generando dataset sintetico"
python scripts/generate_synthetic_dataset.py --n 4000 \
    --output data/raw/cardis_train.csv

echo "[2/5] Construyendo imagen de entrenamiento"
docker compose build trainer

echo "[3/5] Entrenando modelo"
docker compose --profile train run --rm trainer

echo "[4/5] Levantando servicio de inferencia"
docker compose up -d inferer
sleep 12

echo "[5/5] Llamando a /v2/health/ready"
curl -fsS http://localhost:8080/v2/health/ready | tee /tmp/ready.json

cat <<EOF | curl -fsS -X POST -H "Content-Type: application/json" \
    --data-binary @- http://localhost:8080/predict | tee /tmp/predict.json
{
  "patients": [
    {"edad": 65, "altura_cm": 170, "peso_kg": 80, "imc": 27.7,
     "presion_sistolica_1": 145, "presion_sistolica_2": 142, "presion_sistolica_3": 148,
     "colesterol_total": 235, "ldl": 145, "hdl": 42, "glucosa_ayunas": 110,
     "fumador": 1, "antecedentes_familiares": 1, "hospital_origen": "HOSP-001",
     "fecha_visita": "2024-09-15", "notas_medicas": "paciente con diabetes",
     "actividad_fisica": "1h"}
  ]
}
EOF

echo "OK - smoke test completado"
