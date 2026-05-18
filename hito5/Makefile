# Uso: `make help` muestra los objetivos disponibles.

PYTHON ?= python
PIP    ?= pip

.PHONY: help install dev-install lint test data train infer-up infer-down \
        compose-up compose-down k8s-apply k8s-delete docker-build smoke

help:  ## Muestra esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | sort | \
	awk -F':.*?## ' '{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:  ## Instala dependencias de runtime
	$(PIP) install -r requirements/infer.txt

dev-install:  ## Instala dependencias de desarrollo
	$(PIP) install -r requirements/dev.txt

lint:  ## Ejecuta ruff y mypy
	ruff check src tests
	mypy src

test:  ## Ejecuta pytest con cobertura
	pytest --cov=src --cov-report=term-missing

data:  ## Genera dataset sintetico
	$(PYTHON) scripts/generate_synthetic_dataset.py

train:  ## Entrena el modelo localmente
	$(PYTHON) -m src.train

infer-up: ## Lanza la API local con uvicorn
	uvicorn src.api:app --host 0.0.0.0 --port 8080 --reload

docker-build: ## Construye las imagenes de Docker
	docker compose build

compose-up:  ## Despliega el stack con docker-compose
	docker compose up -d minio mlflow inferer

compose-down: ## Para el stack
	docker compose down

compose-train: ## Lanza un job de entrenamiento dentro del stack
	docker compose --profile train run --rm trainer

k8s-apply: ## Aplica los manifests al cluster
	kubectl apply -k kubernetes/

k8s-delete: ## Elimina los recursos del cluster
	kubectl delete -k kubernetes/

smoke: ## Test end-to-end de validacion
	bash scripts/smoke_test.sh

reference: ## Genera reference.parquet y lo sube a MinIO
	$(PYTHON) scripts/generate_reference_dataset.py

docker-build-drift: ## Construye la imagen del drift monitor
	docker build -f docker/Dockerfile.drift -t cardis/drift-monitor:1.0.0 .

kind-load-drift: ## Carga la imagen del drift monitor en el cluster kind
	kind load docker-image cardis/drift-monitor:1.0.0 --name cardis

k8s-drift: docker-build-drift kind-load-drift k8s-apply ## Build + load + deploy del drift monitor
