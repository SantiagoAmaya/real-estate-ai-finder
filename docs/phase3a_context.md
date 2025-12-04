# Contexto para Fase 3A: Computer Vision Analysis

## Estado Actual del Proyecto

### ✅ Fases Completadas

**Fase 1: Data Pipeline & MLOps Foundation**
- DVC con AWS S3 para data versioning
- Scraper de Fotocasa con anti-bot protection (curl_cffi)
- EDA completo con MLflow experiment tracking
- Data quality validation con Great Expectations
- Decisión: 95% de propiedades tienen imágenes, solo 12% mencionan "entrada independiente"

**Fase 2: Query Understanding (NLP)**
- Query Parser: Español → structured filters (JSON)
- Claude API para parsing con Pydantic validation
- Prompt versioning system (v1.0)
- Separación: Direct filters (scraper) vs Indirect filters (análisis)
- Tests: 3/3 passing, 80% coverage
- Integración con scraper funcionando

**Fase 3B: Text Analysis (COMPLETADA)**
- **Sistema flexible**: Detección dinámica de features (ilimitados, no hardcoded)
- **Dual backend**:
  - API (TF-IDF): Rápido para desarrollo (~1-2s/property)
  - Local (sentence-transformers): Preciso para producción (~30s primera vez, ~0.5s cached)
- **Features detectados**: cocina_equipada, luz_natural, entrada_comunidad, terraza, etc.
- **Matching inteligente**: Feature matching + semantic similarity
- **Pipeline end-to-end**: Query → Scrape → Analyze → Match → Rank (FUNCIONAL)

### 📊 Resultados del Test End-to-End

Query: "Local comercial Barcelona con entrada independiente, luz natural, máximo 300 mil euros"
```
✅ Scraped: 36 properties
✅ Analyzed: 20 properties  
✅ Top match score: 0.38 (FAIR)
⚠️  Limitación: Pocos locales mencionan "entrada independiente" en texto
```

**Features más detectados:**
- luz_natural, mucha_luz, patio_trasero_luz_natural
- entrada_comunidad, habitacion_independiente
- cocina_equipada, terraza, parking

---

## 🎯 Fase 3A: Computer Vision Analysis

### Objetivo

Analizar **imágenes** de propiedades para detectar features visuales que NO están en el texto, por ejemplo:
1. **Entrada independiente** - Puerta desde la calle
2. **Luz natural** - Ventanas, ventanales, luz visible
3. **Layout** - Diáfano, espacios separados
4. **Características visuales** - Terraza, parking, estado

### Por qué es Crítico

En las imagenes puede encontrarse gran parte de informacion que no esta descrita o que es muy sutil para ser definida por filtros directos.

### Arquitectura Propuesta
```
Property Images (10-30 per property)
    ↓
Vision Analyzer (Claude Vision API) o Alguna forma local para reducir costos.
    ↓
Visual Features + Confidence Scores
    ↓
Combinar con Text Features
    ↓
Final Scoring: 0.4 * text + 0.4 * vision + 0.2 * semantic
```

### Implementación

**Backend:** Claude Vision API (claude-sonnet-4-20250514)
- ✅ Ya tienes API key
- ✅ Zero setup, alta calidad
- 💰 ~$0.03 por imagen (batch: ~$0.30-0.90 por property)
- ⚡ Rápido: ~2-3s por imagen

**Alternativas futuras:**
- AWS Rekognition (más barato, menos preciso)
- CLIP local (gratis, requiere GPU) Revisar si puedo usar la GPU de mi Macbook pro M3.

### Estructura de Código
```
src/property_analysis/
├── vision_analyzer.py     # NUEVO - Análisis de imágenes
├── text_analyzer.py       # EXISTENTE
├── scorer.py              # NUEVO - Combina text + vision
└── schemas.py             # ACTUALIZAR - añadir VisualFeatures
```

### Features a Detectar (Vision)

Mi idea al alto nivel es obtener una "descripcion" (puede ser un embedding) de las imagenes con lo cual se identificara otro score de similaridad con el query. 

## 🗂️ Datos Actuales

**Propiedades scrapeadas:**
- `data/raw/fotocasa_*.json` - 162 properties total (5 archivos)
- Últimas 36 son de locales Barcelona <300k€
- Todas con imágenes (100%)
- URLs de imágenes listas para análisis

**Embeddings cacheados:**
- `data/cache/embeddings/*.npy` - Embeddings de propiedades analizadas
- Mantener para eficiencia (se regeneran si faltan)

---

## 📝 Próximos Pasos (Fase 3A)

### Semana 1: Vision Analyzer Core
1. [ ] Crear `VisionAnalyzer` class con Claude Vision API
2. [ ] Schema `VisualFeatures` con confidence scores
3. [ ] Batch processing de imágenes (smart selection)
4. [ ] Cache de análisis visual
5. [ ] Tests unitarios

### Semana 2: Integration & Scoring
6. [ ] Integrar vision con text analyzer
7. [ ] Sistema de scoring combinado (text + vision + semantic)
8. [ ] Actualizar pipeline end-to-end
9. [ ] Notebook de evaluación con comparativas

### Semana 3: Optimization & Testing
10. [ ] Optimizar selección de imágenes
11. [ ] Manejo de errores (imágenes corruptas, API failures)
12. [ ] Tests de integración completos
13. [ ] Documentación y demo

---

## 🔧 Tech Stack Actual

**Core:**
- Python 3.10, curl_cffi, BeautifulSoup4
- Anthropic Claude API (Sonnet 4)
- Pydantic v2 para validation
- MLflow para experiment tracking
- DVC + AWS S3 para data versioning

**NLP/Analysis:**
- sentence-transformers (local backend)
- scikit-learn (TF-IDF para API backend)
- numpy para embeddings

**Testing:**
- pytest + pytest-cov
- Rich para CLI output

**Pendiente añadir:**
- Pillow/PIL para manejo de imágenes
- requests para download de imágenes (opcional)

---

## 💡 Decisiones de Diseño Importantes

1. **Features dinámicos** (no hardcoded) - Permite detectar CUALQUIER feature
2. **Dual backend** (api/local) - Flexible para desarrollo vs producción
3. **Separación direct/indirect filters** - Eficiencia en scraping
4. **Caching agresivo** - Embeddings, análisis, imágenes
5. **MLflow tracking** - Todas las decisiones son data-driven

---

## 🎯 Objetivo de Fase 3A

**Input:** Property con imágenes + descripción
**Output:** Combined score con features de texto + visión
**Métrica de éxito:** Aumentar matches relevantes de 20% a 60%+

---

## 📚 Referencias Útiles

- Repo: `/Users/santiagoamaya/Desktop/propAgent/real-estate-ai-finder`
- Docs fases: `/mnt/project/phases.md`
- Tests: `pytest tests/unit/ -v`
- Pipeline: `python scripts/end_to_end_test.py "query"`
- MLflow UI: `mlflow ui --port 5000`

---

## ⚠️ Notas Importantes

- Environment: `conda activate rai` (realestate-ai)
- API Key: En `.env` (ANTHROPIC_API_KEY)
- DVC: Configurado con AWS S3
- Git: Todo trackeado excepto `.env`, `data/raw/`, `data/cache/`