import requests
import time

URL = "http://localhost:8000/predict"

def send_drifted_data():
    payload = {
        "patients": [{
            "edad": 65.0,
            "altura_cm": 170.0,
            "peso_kg": 98.0,
            "imc": 33.9,
            "presion_sistolica_1": 215.0,
            "presion_sistolica_2": 210.0,
            "presion_sistolica_3": 212.0,
            "colesterol_total": 495.0,
            "ldl": 310.0,
            "hdl": 15.0,
            "glucosa_ayunas": 140.0,
            "fumador": 1,
            "antecedentes_familiares": 1,
            "hospital_origen": "General de Ciudad Real",
            "notas_medicas": "Simulación de Drift"
        }]
    }
    try:
        response = requests.post(URL, json=payload, timeout=5)
        if response.status_code == 200:
            print(f"Inyectando drift... Status: 200 | Prob: {response.json()['predictions'][0]['probability']:.2f}")
        else:
            print(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Error de conexión: {e}")

if __name__ == "__main__":
    print("Iniciando inyección de anomalías para activar el retrain webhook...")
    for i in range(50):
        send_drifted_data()
        time.sleep(0.1)
    
    print("\nAtaque finalizado. Ahora toca esperar a que:")
    print("1. El drift-monitor detecte el cambio.")
    print("2. Prometheus dispare la alerta.")
    print("3. El webhook levante el nuevo trainer.")