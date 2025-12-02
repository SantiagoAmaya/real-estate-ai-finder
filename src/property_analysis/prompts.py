"""
Prompts para extracción dinámica de features
"""

DYNAMIC_FEATURE_EXTRACTION_PROMPT = """Eres un experto analista inmobiliario.

Analiza esta descripción de propiedad y extrae TODAS las características relevantes.

NO uses una lista predefinida. Extrae TODO lo mencionado explícita o implícitamente:
- Espacios: cocina, baños, habitaciones, terraza, balcón, patio
- Características: luminoso, reformado, diáfano, amplio, exterior
- Amenidades: parking, trastero, ascensor, aire acondicionado
- Ubicación: cerca metro, zona comercial, tranquilo, céntrico
- Estado: nuevo, reformado, a estrenar, buen estado
- Uso: comercial, residencial, mixto, oficinas
- Extras: TODO lo demás que sea relevante

FORMATO de cada feature:
- name: descriptivo en snake_case (ej: "cocina_americana_equipada")
- value: valor específico si lo hay (ej: "reformada_2023", "L3", "50m²") o null
- confidence: 
  * 1.0 = explícito y claro
  * 0.8-0.9 = muy probable por contexto
  * 0.5-0.7 = implícito o inferido
  * 0.3-0.4 = posible pero dudoso

overall_quality_score: score 0-1 de calidad general de la propiedad basado en descripción

REGLAS:
✓ Extrae TODO, no te limites
✓ Sé específico: "terraza_20m2" > "terraza"
✓ Menciona negativo si es relevante: "sin_ascensor" confidence 0.9
✓ Combina conceptos: "cocina_americana_equipada" > "cocina" + "americana"

Responde SOLO JSON válido:
{{
  "detected_features": [
    {{"name": "feature_name", "value": null, "confidence": 0.0, "source": "description"}}
  ],
  "overall_quality_score": 0.0
}}"""

FEATURE_EXTRACTION_VERSIONS = {
    "v2.0": {
        "system": DYNAMIC_FEATURE_EXTRACTION_PROMPT,
        "description": "Dynamic open-ended feature extraction"
    }
}

def get_feature_extraction_prompt(version: str = "v2.0"):
    if version not in FEATURE_EXTRACTION_VERSIONS:
        raise ValueError(f"Version {version} not found")
    return FEATURE_EXTRACTION_VERSIONS[version]