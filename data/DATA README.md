# Data Directory Documentation

## Overview

Este directorio contiene todos los datos del proyecto, organizados por etapa del pipeline y versionados con DVC (Data Version Control) almacenado en AWS S3.

## Directory Structure

```
data/
├── raw/                    # Datos crudos del scraper (DVC tracked)
│   ├── fotocasa_YYYYMMDD_HHMMSS.json
│   └── ...
├── processed/              # Datos procesados/limpios (DVC tracked)
│   └── .gitkeep
├── cache/                  # Cache temporal (NO versionado)
│   └── .gitkeep
└── images/                 # Imágenes descargadas (DVC tracked)
    └── {property_id}/
        ├── 0.jpg
        ├── 1.jpg
        └── ...
```

## DVC Configuration

### Remote Storage
- **Type**: AWS S3
- **Bucket**: `mlops-real-estate-santi`
- **Region**: `eu-west-1` (Ireland)
- **Path prefix**: `dvc-storage/`

### Current DVC Status
```bash
# Ver status actual
dvc status

# Ver remote configurado
dvc remote list
```

## Working with Data

### 1. First Time Setup

Si eres un nuevo colaborador o estás clonando el proyecto:

```bash
# 1. Clonar el repositorio
git clone <repo-url>
cd real-estate-mlops

# 2. Configurar AWS credentials (necesario para DVC)
aws configure
# Ingresa tu Access Key ID y Secret Access Key

# 3. Descargar todos los datos
dvc pull

# Ahora tienes todos los datos en data/raw/
```

### 2. Adding New Data

Después de ejecutar el scraper y generar nuevos datos:

```bash
# 1. Ejecutar scraping (genera nuevo JSON en data/raw/)
python scripts/run_scraping.py --location barcelona-capital --max-pages 5

# 2. Añadir el directorio completo a DVC
dvc add data/raw/

# 3. Subir datos a S3
dvc push

# 4. Commit el archivo .dvc a git
git add data/raw.dvc .gitignore
git commit -m "Add new scraped data - YYYY-MM-DD"
git push
```

**⚠️ IMPORTANTE**: Los datos reales NO van a git, solo el archivo `.dvc` que contiene los hashes.

### 3. Updating to Latest Data

Cuando otro miembro del equipo añade nuevos datos:

```bash
# 1. Pull cambios de git (obtiene el .dvc actualizado)
git pull

# 2. Pull datos de S3 (obtiene los datos reales)
dvc pull

# Ahora tienes los datos más recientes
```

### 4. Working with Specific Versions

DVC permite volver a versiones anteriores de datos:

```bash
# Ver historial de versiones
git log --oneline data/raw.dvc

# Ejemplo de output:
# a1b2c3d Add new scraped data - 2025-11-25
# e4f5g6h Add new scraped data - 2025-11-23
# i7j8k9l Add initial data - 2025-11-20

# Volver a una versión específica
git checkout e4f5g6h data/raw.dvc
dvc checkout

# Los archivos en data/raw/ ahora son de esa versión

# Volver a la última versión
git checkout main data/raw.dvc
dvc checkout
```

## Data Versioning Best Practices

### Naming Convention

Los archivos de datos raw siguen este formato:
```
fotocasa_YYYYMMDD_HHMMSS.json
```

Ejemplo: `fotocasa_20251123_154917.json`
- Timestamp permite identificar cuándo se scrapeó
- Útil para análisis temporal del mercado
- Evita conflictos de nombres

### When to Create New Data Versions

Crea una nueva versión de datos cuando:
- ✅ Scrapeaste nuevas propiedades (nueva ejecución del scraper)
- ✅ Cambiaste los filtros de scraping significativamente
- ✅ Actualizaste el scraper (nueva versión que extrae más campos)
- ✅ Pasó suficiente tiempo (ej: scraping semanal)

NO crees nueva versión para:
- ❌ Cambios menores en formato (usa processing pipeline)
- ❌ Correcciones de bugs que no afectan los datos
- ❌ Reorganización de archivos sin cambios de contenido

### Data Size Management

Monitorea el tamaño de tus datos:

```bash
# Ver tamaño de data/raw/
du -sh data/raw/

# Ver tamaño en S3
aws s3 ls s3://mlops-real-estate-santi/dvc-storage/ --recursive --summarize
```

**Costos estimados S3**:
- JSON data: ~10KB por propiedad → 500 propiedades = ~5MB
- Imágenes: ~200KB por imagen → 500 props × 5 imgs = ~500MB
- **Costo mensual estimado**: $0.10 - 0.30 (dentro de free tier)

## Data Quality Checks

Antes de versionar nuevos datos, siempre verifica:

### 1. Basic Checks
```bash
# Ver metadata del último archivo
python -c "
import json
with open('data/raw/fotocasa_latest.json') as f:
    data = json.load(f)
    print('Properties:', len(data['properties']))
    print('Quality:', data['metadata']['data_quality'])
"
```

### 2. Automated Validation (Coming in Phase 1c)
```bash
# Ejecutar validación completa
python scripts/validate_data.py data/raw/fotocasa_latest.json
```

## Common Issues & Solutions

### Issue 1: "dvc pull" fails with authentication error

**Solución**:
```bash
# Verificar AWS credentials
aws s3 ls s3://mlops-real-estate-santi/

# Si falla, reconfigurar
aws configure
```

### Issue 2: DVC says "already in cache" but files missing

**Solución**:
```bash
# Forzar checkout desde cache
dvc checkout --force
```

### Issue 3: Accidentally committed data to git

**Solución**:
```bash
# Remover del historial (ANTES de push)
git rm -r --cached data/raw/
git commit -m "Remove data from git (should be in DVC only)"

# Añadir correctamente a DVC
dvc add data/raw/
git add data/raw.dvc .gitignore
git commit -m "Add data to DVC properly"
```

### Issue 4: S3 storage costs increasing

**Solución**:
```bash
# Ver versiones antiguas en DVC
dvc gc --workspace --cloud

# Limpiar versiones antiguas (cuidado!)
dvc gc --workspace --cloud --force
```

## Data Schema

Ver documentación detallada del esquema en: [`docs/data_schema.md`](../docs/data_schema.md)

Quick reference de campos principales:
- `id`: Identificador único de Fotocasa
- `price`: Precio en EUR
- `location`: "Barrio, Distrito, Municipio"
- `size_m2`: Metros cuadrados (opcional)
- `rooms`: Número de habitaciones (opcional)
- `description`: Descripción completa en español
- `images`: Array de URLs de imágenes

## Next Steps

Esta es la Fase 1 del proyecto. Los próximos pasos son:

1. **Fase 1b** (En Progreso): EDA + MLflow setup
2. **Fase 1c** (Próximo): Data quality monitoring con Great Expectations
3. **Fase 2**: Query Parser (NLP)
4. **Fase 3**: Property Analysis Models (CV + NLP)

## Resources

- [DVC Documentation](https://dvc.org/doc)
- [AWS S3 Pricing](https://aws.amazon.com/s3/pricing/)
- [Project Phases](../docs/phases.md)

---

**Last Updated**: 2025-11-23  
**Maintainer**: Santiago (PhD Student - MARL & MLOps)