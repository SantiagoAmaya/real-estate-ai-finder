"""
Tests para query parser
"""
import pytest
import os
from src.query_parser.parser import QueryParser
from src.query_parser.schemas import ParsedQuery

# Skip si no hay API key
skip_if_no_key = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set"
)

@skip_if_no_key
def test_parse_simple_query():
    """Test query básica con precio"""
    parser = QueryParser()
    result = parser.parse("Piso en Barcelona máximo 300k")
    
    assert result.original_query == "Piso en Barcelona máximo 300k"
    assert result.direct_filters.location in ["barcelona-capital", "barcelona"]
    assert result.direct_filters.max_price == 300000
    assert result.confidence > 0.5

@skip_if_no_key
def test_parse_complex_query():
    """Test query con filtros indirectos"""
    parser = QueryParser()
    result = parser.parse(
        "Local comercial Barcelona con entrada independiente y luz natural, máx 250k"
    )
    
    assert result.direct_filters.property_type == "local"
    assert result.direct_filters.max_price == 250000
    assert result.indirect_filters.entrance_type == "independent"
    assert result.indirect_filters.natural_light in ["required", "preferred"]

def test_parser_fallback():
    """Test que parser no falla con query inválida"""
    parser = QueryParser()
    result = parser.parse("")
    
    assert isinstance(result, ParsedQuery)
    assert result.confidence == 0.0