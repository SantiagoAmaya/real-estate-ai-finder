"""
Tests para text analyzer
"""
import pytest
import os
from src.property_analysis.text_analyzer import PropertyTextAnalyzer

skip_if_no_key = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set"
)

@skip_if_no_key
def test_analyze_with_independent_entrance():
    """Test detección de entrada independiente"""
    analyzer = PropertyTextAnalyzer()
    
    description = """
    Espacioso local comercial con entrada independiente desde la calle.
    Ideal para cualquier tipo de negocio.
    """
    
    result = analyzer.analyze("test_001", description)
    
    assert result.features.independent_entrance > 0.7
    assert result.features.commercial_use > 0.7
    assert result.features.confidence > 0

@skip_if_no_key
def test_analyze_with_natural_light():
    """Test detección de luz natural"""
    analyzer = PropertyTextAnalyzer()
    
    description = """
    Piso muy luminoso con grandes ventanales orientados al sur.
    Luz natural durante todo el día.
    """
    
    result = analyzer.analyze("test_002", description)
    
    assert result.features.natural_light > 0.7
    assert result.features.confidence > 0

def test_analyze_empty_description():
    """Test con descripción vacía"""
    analyzer = PropertyTextAnalyzer()
    
    result = analyzer.analyze("test_003", "")
    
    assert result.features.confidence == 0.0
    assert result.features.independent_entrance == 0.0

@skip_if_no_key
def test_get_detected_features():
    """Test filtrado de features detectados"""
    analyzer = PropertyTextAnalyzer()
    
    description = "Local con terraza y parking incluido"
    result = analyzer.analyze("test_004", description)
    
    detected = result.features.get_detected_features(threshold=0.5)
    
    # Solo debe incluir features con score >= 0.5
    for feature, score in detected.items():
        assert score >= 0.5