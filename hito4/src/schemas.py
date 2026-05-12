"""Esquemas Pydantic alineados con el Open Inference Protocol (V2).

KServe / NVIDIA Triton / Seldon Core utilizan este protocolo para
unificar la interfaz de los servidores de inferencia. Modelar nuestras
peticiones siguiendo este estandar permite reemplazar el backend de
servido sin modificar a los clientes (la interfaz clinica de CARDIS).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
# Open Inference Protocol (subset)
# -----------------------------------------------------------------------------
class InferenceTensor(BaseModel):
    name: str = Field(..., description="Nombre del tensor de entrada/salida")
    shape: List[int]
    datatype: str = Field(default="FP32")
    data: List[Any]
    parameters: Optional[Dict[str, Any]] = None


class InferenceRequest(BaseModel):
    """Peticion compatible con KServe v2."""

    id: Optional[str] = Field(default=None, description="Trace identifier")
    parameters: Optional[Dict[str, Any]] = None
    inputs: List[InferenceTensor]


class InferenceResponse(BaseModel):
    id: Optional[str] = None
    model_name: str = "cardis-lightgbm"
    model_version: str = "1.0.0"
    parameters: Optional[Dict[str, Any]] = None
    outputs: List[InferenceTensor]


# -----------------------------------------------------------------------------
# Esquema de alto nivel para clientes que prefieren JSON tabular
# -----------------------------------------------------------------------------
class PatientRecord(BaseModel):
    """Representacion de un paciente en el dominio CARDIS.

    Solo se exponen las variables observables en una historia clinica
    real; los identificadores y campos no clinicos (id, talla_zapato,
    codigo_postal) no se incluyen para evitar discriminacion.
    """

    edad: float = Field(..., ge=0, le=120)
    altura_cm: float = Field(..., ge=120, le=220)
    peso_kg: float = Field(..., ge=20, le=250)
    imc: Optional[float] = Field(default=None, ge=10, le=70)
    presion_sistolica_1: float
    presion_sistolica_2: float
    presion_sistolica_3: float
    colesterol_total: float
    ldl: float
    hdl: float
    glucosa_ayunas: Optional[float] = None
    fumador: int = Field(..., ge=0, le=1)
    antecedentes_familiares: int = Field(..., ge=0, le=1)
    hospital_origen: str
    fecha_visita: Optional[str] = None
    notas_medicas: Optional[str] = None
    actividad_fisica: Optional[str] = None


class PredictRequest(BaseModel):
    patients: List[PatientRecord]


class PredictResult(BaseModel):
    probability: float = Field(..., ge=0.0, le=1.0)
    label: int
    threshold: float
    age_warning: bool = Field(
        default=False,
        description="True si el paciente es < 56 anios (sesgo conocido).",
    )


class PredictResponse(BaseModel):
    model_name: str = "cardis-lightgbm"
    model_version: str = "1.0.0"
    threshold: float
    predictions: List[PredictResult]


# -----------------------------------------------------------------------------
# Health / metadata
# -----------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    model_name: str = "cardis-lightgbm"
    model_version: str = "1.0.0"


class ModelMetadata(BaseModel):
    name: str = "cardis-lightgbm"
    version: str = "1.0.0"
    platform: str = "lightgbm"
    inputs: List[InferenceTensor] = []
    outputs: List[InferenceTensor] = []
    threshold: float
    feature_names: List[str]
