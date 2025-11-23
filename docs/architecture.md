# System Architecture

## Overview

Real Estate AI Finder uses multi-modal AI to find properties matching complex natural language requirements in Barcelona.

## Components

### 1. Data Pipeline
- **Scraping:** BeautifulSoup + Requests for Idealista/Fotocasa
- **Storage:** Local JSON files, versioned with DVC
- **Validation:** Pydantic models for data quality
- **Orchestration:** Prefect for automated collection

### 2. ML Models
- **Query Parser:** LLM-based NLP to parse Spanish queries
- **Property Analyzer:** Multi-modal analysis (text + images)
- **Ranking System:** Score properties against requirements

### 3. API & Interface
- **API:** FastAPI REST endpoints
- **UI:** Simple web interface (optional)

### 4. MLOps Infrastructure
- **Experiments:** MLflow for tracking
- **Orchestration:** Prefect for workflows
- **CI/CD:** GitHub Actions
- **Monitoring:** Prometheus + Grafana (Phase 5)

## Data Flow

```
Scraper → Raw Data → DVC → Validation → Clean Data
   ↓
MLflow (log metrics)
   ↓
Query → Parser → Analyzer → Ranker → Results
```

## Technology Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Environment | Conda | Industry standard for data science |
| Data Versioning | DVC | Git for data |
| Experiment Tracking | MLflow | Model registry + metrics |
| Orchestration | Prefect | Modern workflow tool |
| API | FastAPI | Fast, modern, async |
| Testing | Pytest | Standard Python testing |
| Code Quality | Black, Flake8 | Consistent code style |

## Development Workflow

1. **Data Collection:** Scraper → Raw Data → DVC commit
2. **Exploration:** Jupyter notebooks with MLflow tracking
3. **Development:** Feature branch → Tests → Pre-commit → PR
4. **Deployment:** Docker → AWS (Phase 5)
