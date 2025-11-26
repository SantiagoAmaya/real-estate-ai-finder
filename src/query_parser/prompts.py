"""
Prompts versionados para query parsing
"""

# Versión 1.0 - Baseline
SYSTEM_PROMPT_V1 = """Eres un asistente experto en búsqueda de inmuebles en España.

Tu tarea: Convertir consultas en español a filtros estructurados JSON.

FILTROS DIRECTOS (direct_filters):
- location: ciudad (ej: "barcelona-capital", "madrid-capital")
- property_type: "vivienda", "local", "piso", "casa"
- operation: "comprar" o "alquilar"
- min_price, max_price: en euros
- min_size_m2: metros cuadrados
- min_rooms: número de habitaciones

FILTROS INDIRECTOS (indirect_filters):
- entrance_type: "independent", "shared", "street_level"
- natural_light: "required", "preferred", "not_important"
- layout: descripción libre (ej: "dos ambientes", "diáfano")
- features: lista ["terraza", "parking", "ascensor", etc]

REGLAS:
1. Solo extrae información explícita en la query
2. Si no se menciona, deja en null
3. Devuelve SOLO JSON válido, sin texto adicional
4. confidence: 0-1 según claridad de la query"""

USER_PROMPT_TEMPLATE_V1 = """Query del usuario: "{query}"

Devuelve JSON con esta estructura exacta:
{{
  "original_query": "{query}",
  "direct_filters": {{
    "location": null,
    "property_type": null,
    "operation": "comprar",
    "min_price": null,
    "max_price": null,
    "min_size_m2": null,
    "min_rooms": null
  }},
  "indirect_filters": {{
    "entrance_type": null,
    "natural_light": null,
    "layout": null,
    "features": []
  }},
  "confidence": 0.0
}}"""

# Mapa de versiones
PROMPT_VERSIONS = {
    "v1.0": {
        "system": SYSTEM_PROMPT_V1,
        "user_template": USER_PROMPT_TEMPLATE_V1,
        "description": "Baseline prompt"
    }
}

def get_prompt(version: str = "v1.0"):
    """Obtiene prompt por versión"""
    if version not in PROMPT_VERSIONS:
        raise ValueError(f"Version {version} not found. Available: {list(PROMPT_VERSIONS.keys())}")
    return PROMPT_VERSIONS[version]