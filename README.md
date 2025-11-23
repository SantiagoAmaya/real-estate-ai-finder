# 🏠 Real Estate AI Finder

An MLOps project for intelligent real estate search in Barcelona using multi-modal AI.

## 🎯 Project Goal

Find properties with complex, nuanced requirements that traditional filters cannot handle.

**Example query:**
> "Quiero encontrar locales en Barcelona que tengan al menos dos estancias separadas, donde una se pueda alquilar a un comercio, y otra pueda ser acondicionada como una vivienda. La parte de vivienda preferiblemente que tenga luz, y lo ideal sería que se pueda entrar en la 'vivienda' sin tener que pasar por el local."

## 🚀 Quick Start

### Setup

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate realestate-ai

# Setup DVC and pre-commit (one-time)
bash scripts/setup_dvc.sh
bash scripts/setup_precommit.sh

# Install package in development mode
pip install -e .
```

### Usage

```bash
# Activate environment (always do this first!)
conda activate realestate-ai

# Run scraper
make scrape

# Run tests
make test

# Start MLflow UI
make mlflow

# Start Jupyter notebook
make notebook
```

## 📁 Project Structure

```
real-estate-ai-finder/
├── data/              # Data storage (gitignored, tracked by DVC)
├── src/               # Source code
│   ├── data/         # Data collection and processing
│   ├── models/       # ML models
│   ├── api/          # FastAPI application
│   └── pipelines/    # MLOps pipelines
├── tests/            # Unit and integration tests
├── notebooks/        # Jupyter notebooks for exploration
├── config/           # Configuration files
└── docs/             # Documentation
```

## 🛠️ Tech Stack

- **Environment:** Conda
- **Data:** DVC for versioning, BeautifulSoup for scraping
- **MLOps:** MLflow (experiments), Prefect (orchestration)
- **ML:** LLMs for NLP, CV for images
- **API:** FastAPI
- **Testing:** Pytest
- **Code Quality:** Black, Flake8, MyPy, Pre-commit

## 📋 Development Phases

- [x] Phase 0: Project setup
- [ ] Phase 1: Data pipeline (Weeks 1-3)
- [ ] Phase 2: Query understanding (Weeks 4-5)
- [ ] Phase 3: Property analysis (Weeks 6-8)
- [ ] Phase 4: End-to-end pipeline (Weeks 9-10)
- [ ] Phase 5: Deployment & monitoring (Weeks 11-12)
- [ ] Phase 6: Documentation & polish (Weeks 13-14)

## 🧪 Testing

```bash
# Run all tests
pytest

# With coverage report
pytest --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html
```

## 📝 Common Commands

```bash
# Format code
make format

# Lint code
make lint

# Clean generated files
make clean

# Update environment
conda env update -f environment.yml --prune
```

## 📚 Documentation

- [Architecture](docs/architecture.md)
- [Phase 1 Plan](docs/phase1_plan.md)
- [Data Schema](docs/data_schema.md)

## 🤝 Contributing

This is a learning project. The main workflow is:
1. Create a feature branch
2. Make changes
3. Run tests: `make test`
4. Format code: `make format`
5. Commit (pre-commit hooks will run)
6. Push and open PR

## Data Management

This project uses DVC (Data Version Control) for data versioning with AWS S3 storage.

### Quick Start
```bash
# Clone repository
git clone <repo-url>
cd real-estate-ai-finder

# Pull data from S3
dvc pull

# Data is now in data/raw/
```

See [data/README.md](data/README.md) for detailed data documentation.
## 📝 License

MIT
