.PHONY: help setup test lint format clean scrape mlflow prefect

CONDA_ENV = realestate-ai

help:
	@echo "Available commands:"
	@echo "  make setup        - Setup development environment"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linters"
	@echo "  make format       - Format code"
	@echo "  make scrape       - Run scraper"
	@echo "  make mlflow       - Start MLflow UI"
	@echo "  make prefect      - Start Prefect UI"
	@echo "  make notebook     - Start Jupyter notebook"
	@echo "  make clean        - Clean generated files"
	@echo ""
	@echo "⚠️  Remember to activate conda environment first:"
	@echo "     conda activate $(CONDA_ENV)"

setup:
	@echo "Run these commands manually:"
	@echo "  conda env create -f environment.yml"
	@echo "  conda activate $(CONDA_ENV)"
	@echo "  bash scripts/setup_dvc.sh"
	@echo "  bash scripts/setup_precommit.sh"

test:
	pytest

lint:
	flake8 src tests
	mypy src

format:
	black src tests
	isort src tests

scrape:
	python src/data/scraper.py

mlflow:
	mlflow ui --host 0.0.0.0 --port 5000

prefect:
	prefect server start

notebook:
	jupyter notebook notebooks/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov dist build *.egg-info

install-package:
	pip install -e .

##@ Data Quality

data-validate: ## Valida calidad de todos los datos
	python scripts/validate_data.py

data-validate-latest: ## Valida solo el archivo más reciente
	python scripts/validate_data.py $$(ls -t data/raw/fotocasa_*.json | head -1)

##@ Property Analysis

analyze-property: ## Analiza una propiedad específica (make analyze-property ID=12345)
	python scripts/analyze_property.py $(ID)

analyze-all: ## Analiza todas las propiedades
	python scripts/analyze_property.py all
