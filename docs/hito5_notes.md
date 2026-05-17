# Notas de desarrollo — Hito 5

## 2025-05-17 — Bloque 3 cerrado: cluster kind + stack observabilidad

### Resumen

Se desplegó con éxito el cluster `kind` local `cardis` y se completó el
**Bloque 3** del Hito 5 (stack de observabilidad).

### Componentes desplegados

| Componente        | Imagen                                     | Estado     |
|-------------------|--------------------------------------------|------------|
| MinIO             | minio/minio:RELEASE.2024-09-22T00-33-43Z  | Running    |
| MLflow            | cardis/mlflow:1.0.0                        | Running    |
| Inferer (×2)      | cardis/inferer:1.0.0                       | Running    |
| Prometheus        | prom/prometheus:v2.53.3                    | Running    |
| Alertmanager      | prom/alertmanager:v0.27.0                  | Running    |
| Grafana           | grafana/grafana:11.4.0                     | Running    |

### Verificación

- Prometheus targets: `cardis-pods` (2 instancias inferer) → **UP** ✓
- 38 inferencias exitosas; métricas `cardis_inference_duration_seconds_count=38`,
  `cardis_prediction_probability_count=38` visibles ✓
- Grafana datasource `cardis-prometheus` → **OK** ✓
- Dashboard `CARDIS — Observabilidad` (UID `cardis-main`, 15 paneles) provisionado ✓
- Alertmanager `/-/healthy` → **OK** ✓

### Problemas encontrados

1. `docker compose build` fallaba (campo `depends_on.required` no soportado en v3.9)
   → solución: `docker build` directo.
2. Disco host al 100% al arrancar Grafana → `No space left on device` al crear
   `/var/lib/grafana/plugins`. Solución: `docker builder prune -f` (liberó 69 MB).

### Ficheros nuevos en `hito5/`

- `kind-cluster.yaml` — configuración del cluster kind
- `kubernetes/prometheus.yaml` — ServiceAccount + RBAC + ConfigMap + Deployment + Service
- `kubernetes/alertmanager.yaml` — ConfigMap + Deployment + Service
- `kubernetes/grafana.yaml` — Secret + Deployment (emptyDir) + Service
- `kubernetes/grafana-provisioning.yaml` — ConfigMaps datasource + dashboard-provider
- `kubernetes/grafana-dashboards.yaml` — ConfigMap con dashboard JSON (15 paneles)
