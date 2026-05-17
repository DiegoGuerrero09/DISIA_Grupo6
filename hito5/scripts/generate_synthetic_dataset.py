"""Genera un dataset sintetico minimo compatible con el formato CARDIS.

Sirve para probar el pipeline end-to-end sin necesidad de descargar
los datos originales de Kaggle. Replica los rangos descritos en la
seccion 2.3 del Hito 2.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def synth(n: int = 24_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    edad = rng.integers(30, 85, n)

    altura = rng.normal(170, 9, n).clip(140, 200)
    peso = rng.normal(75, 14, n).clip(40, 130)
    imc = peso / (altura / 100) ** 2

    p1 = rng.normal(120 + (edad - 50) * 0.4, 15, n).clip(84, 220)
    p2 = p1 + rng.normal(0, 5, n)
    p3 = p1 + rng.normal(0, 5, n)

    ldl = rng.normal(125, 30, n).clip(40, 230)
    hdl = rng.normal(45, 8, n).clip(28, 90)
    col = ldl + hdl + rng.normal(60, 15, n)
    glucosa = rng.normal(95 + (edad - 50) * 0.3, 10, n).clip(58, 140)
    glucosa = np.where(rng.random(n) < (1 - (edad >= 56)) * 0.85, np.nan, glucosa)

    fumador = rng.integers(0, 2, n).astype(float)
    fumador = np.where(rng.random(n) < 0.09, np.nan, fumador)
    antec = rng.integers(0, 2, n)

    hospital = rng.choice([f"HOSP-{i:03d}" for i in range(1, 21)], n)
    cp = rng.choice([f"{rng.integers(1000, 52999)}" for _ in range(791)], n)

    fechas = pd.to_datetime("2017-01-01") + pd.to_timedelta(
        rng.integers(0, 365 * 5, n), unit="D"
    )

    actividades = rng.choice(
        ["2.5h", "1h", "0.5h", "5 horas", "poco", "3,5 horas", "0", "4h"], n
    )
    notas_pool = [
        "paciente normal",
        "diabetes tipo 2",
        "infarto previo, dolor pecho",
        "fumador habitual, sedentarismo",
        "fatiga ocasional, mareo",
        "controlado, estable",
        "angina de esfuerzo",
        None,
    ]
    notas = rng.choice(notas_pool, n, p=[0.30, 0.10, 0.05, 0.10, 0.05, 0.10, 0.05, 0.25])

    score = (
        (edad - 30) * 0.025
        + (ldl > 130) * 0.4
        + (p1 > 140) * 0.4
        + (np.nan_to_num(glucosa, nan=95) > 110) * 0.3
        + np.nan_to_num(fumador, nan=0) * 0.4
        + antec * 0.3
        + ((notas == "diabetes tipo 2") | (notas == "infarto previo, dolor pecho")).astype(int) * 0.5
    )
    proba = 1 / (1 + np.exp(-(score - 1.6)))
    riesgo = (rng.random(n) < proba).astype(int)

    return pd.DataFrame(
        {
            "id": np.arange(n),
            "edad": edad,
            "altura_cm": altura.round(1),
            "peso_kg": peso.round(1),
            "imc": imc.round(1),
            "presion_sistolica_1": p1.round(1),
            "presion_sistolica_2": p2.round(1),
            "presion_sistolica_3": p3.round(1),
            "colesterol_total": col.round(1),
            "ldl": ldl.round(1),
            "hdl": hdl.round(1),
            "glucosa_ayunas": glucosa.round(1),
            "fumador": fumador,
            "antecedentes_familiares": antec,
            "hospital_origen": hospital,
            "codigo_postal": cp,
            "fecha_visita": fechas,
            "notas_medicas": notas,
            "actividad_fisica": actividades,
            "talla_zapato": rng.integers(34, 46, n),
            "riesgo_cv": riesgo,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=24_000)
    parser.add_argument(
        "--output", type=Path, default=Path("data/raw/cardis_train.csv")
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = synth(args.n, args.seed)
    df.to_csv(args.output, index=False)
    print(f"Dataset sintetico escrito en {args.output}: {df.shape}")


if __name__ == "__main__":
    main()
