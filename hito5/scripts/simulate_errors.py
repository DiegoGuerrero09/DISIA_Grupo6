import requests

URL = "http://localhost:8080/v2/models/cardis-lightgbm/infer"

payload = {
    "id": "test",
    "inputs": [
        {
            "name": "patients_json",
            "shape": [1],
            "data": [{}]
        }
    ]
}

while True:
    r = requests.post(URL, json=payload)
    print(r.status_code, r.text)
