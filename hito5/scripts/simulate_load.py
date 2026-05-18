import requests
import time
import random

URL = "http://localhost:8080/predict" 

def send_request():
    payload = {
        "patients": [{
            "edad": random.uniform(56, 80),
            "altura_cm": 175.0,
            "peso_kg": 80.0,
            "imc": 26.1,
            "presion_sistolica_1": random.uniform(120, 140),
            "presion_sistolica_2": 130.0,
            "presion_sistolica_3": 130.0,
            "colesterol_total": random.uniform(180, 220),
            "ldl": 110.0,
            "hdl": 50.0,
            "glucosa_ayunas": 95.0,
            "fumador": random.choice([0, 1]),
            "antecedentes_familiares": 0,
            "hospital_origen": "General de Ciudad Real",
            "notas_medicas": "Simulación de carga normal"
        }]
    }
    try:
        response = requests.post(URL, json=payload, timeout=5)
        if response.status_code == 200:
            res_json = response.json()
            prob = res_json['predictions'][0]['probability']
            print(f"Status: 200 | Probabilidad: {prob:.2f}")
        else:
            print(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Fallo de conexión: {e}")

if __name__ == "__main__":
    print("Atacando endpoint /predict...")
    while True:
        send_request()