"""
Pydantic schemas para query parsing
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Literal

class DirectFilters(BaseModel):
    """Filtros que van directo al scraper"""
    location: Optional[str] = Field(None, description="barcelona-capital, madrid-capital, etc")
    property_type: Optional[Literal["vivienda", "local", "piso", "casa"]] = None
    operation: Optional[Literal["comprar", "alquilar"]] = "comprar"
    min_price: Optional[int] = None
    max_price: Optional[int] = None
    min_size_m2: Optional[int] = None
    min_rooms: Optional[int] = None

class IndirectFilters(BaseModel):
    """Filtros que requieren análisis (NLP/CV)"""
    entrance_type: Optional[Literal["independent", "shared", "street_level"]] = None
    natural_light: Optional[Literal["required", "preferred", "not_important"]] = None
    layout: Optional[str] = Field(None, description="ej: 'dos ambientes', 'diáfano'")
    features: Optional[List[str]] = Field(default_factory=list, description="terraza, parking, etc")
    
class ParsedQuery(BaseModel):
    """Query completa parseada"""
    original_query: str
    direct_filters: DirectFilters
    indirect_filters: IndirectFilters
    confidence: float = Field(ge=0, le=1, description="Confianza del parsing")