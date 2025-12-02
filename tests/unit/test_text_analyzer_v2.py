"""
Tests para text analyzer v2.0
"""
import pytest
import os
from src.property_analysis.text_analyzer import PropertyTextAnalyzer
from src.property_analysis.schemas import QueryRequirement

skip_if_no_key = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set"
)

@skip_if_no_key
def test_dynamic_feature_extraction():
    """Test que extrae features no predefinidos"""
    analyzer = PropertyTextAnalyzer()
    
    description = """
    Magnífico local con cocina industrial completamente equipada.
    Ubicado a 2 minutos del metro L3.
    Incluye terraza privada de 25m² y trastero.
    """
    
    result = analyzer.analyze("test_001", description, generate_embedding=False)
    
    # Debe detectar features dinámicos
    assert len(result.detected_features) > 0
    
    # Verificar que detectó algo relacionado con cocina
    cocina_features = [f for f in result.detected_features if "cocina" in f.name.lower()]
    assert len(cocina_features) > 0
    
    # Verificar confidence
    assert result.overall_quality_score > 0

@skip_if_no_key
def test_embedding_generation():
    """Test generación de embeddings"""
    analyzer = PropertyTextAnalyzer()
    
    description = "Local luminoso con entrada independiente"
    result = analyzer.analyze("test_002", description, generate_embedding=True)
    
    # Debe tener embedding
    assert result.text_embedding is not None
    assert len(result.text_embedding) == 384  # MiniLM dims
    
    # Embedding debe ser numpy-compatible
    embedding_array = result.get_embedding_array()
    assert embedding_array is not None
    assert embedding_array.shape == (384,)

@skip_if_no_key
def test_semantic_matching():
    """Test matching semántico"""
    analyzer = PropertyTextAnalyzer()
    
    # Analizar propiedad
    description = "Local con cocina equipada y terraza de 20m²"
    analysis = analyzer.analyze("test_003", description, generate_embedding=True)
    
    # Match contra query
    query = "Busco local con cocina y espacio exterior"
    requirements = [
        QueryRequirement(feature_name="cocina_equipada", importance=1.0),
        QueryRequirement(feature_name="terraza", importance=0.8)
    ]
    
    match_result = analyzer.match_against_query(
        analysis,
        query,
        requirements
    )
    
    # Debe tener score razonable
    assert match_result.final_score > 0.3
    assert match_result.semantic_similarity_score > 0

@skip_if_no_key
def test_fuzzy_feature_matching():
    """Test que sinónimos funcionan"""
    analyzer = PropertyTextAnalyzer()
    
    # Propiedad menciona "cocina americana"
    description = "Local con cocina americana integrada"
    analysis = analyzer.analyze("test_004", description, generate_embedding=False)
    
    # Usuario busca "cocina abierta" (sinónimo)
    requirements = [QueryRequirement(feature_name="cocina_abierta", importance=1.0)]
    
    match_result = analyzer.match_against_query(analysis, "cocina", requirements)
    
    # Debe matchear por similitud semántica
    assert match_result.feature_match_score > 0 or len(match_result.matched_features) > 0