"""
Schemas flexibles para análisis de propiedades
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np

class DetectedFeature(BaseModel):
    """Feature detectado dinámicamente"""
    name: str = Field(..., description="Nombre en snake_case: cocina_equipada, metro_cercano")
    value: Optional[str] = Field(None, description="Valor específico: 'L3', '20m²', 'present'")
    confidence: float = Field(..., ge=0, le=1, description="Confianza 0-1")
    source: str = Field(default="description", description="Origen: description/image/structured")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "terraza_privada",
                "value": "20m²",
                "confidence": 0.95,
                "source": "description"
            }
        }

class PropertyAnalysis(BaseModel):
    """Análisis completo y flexible de una propiedad"""
    property_id: str
    
    # Features dinámicos
    detected_features: List[DetectedFeature] = Field(default_factory=list)
    
    # Embedding para búsqueda semántica (768 dims - sentence-transformers)
    text_embedding: Optional[List[float]] = Field(None, exclude=True)  # No serializar por defecto
    
    # Metadata
    description_length: int = 0
    overall_quality_score: float = Field(0.0, ge=0, le=1)
    analysis_version: str = "v2.0"
    
    def get_feature(self, name: str) -> Optional[DetectedFeature]:
        """Busca feature por nombre exacto"""
        for feature in self.detected_features:
            if feature.name.lower() == name.lower():
                return feature
        return None
    
    def has_feature(self, name: str, min_confidence: float = 0.5) -> bool:
        """Check si tiene feature con confianza mínima"""
        feature = self.get_feature(name)
        return feature is not None and feature.confidence >= min_confidence
    
    def get_features_dict(self) -> Dict[str, float]:
        """Features como dict {nombre: confidence}"""
        return {f.name: f.confidence for f in self.detected_features}
    
    def get_embedding_array(self) -> Optional[np.ndarray]:
        """Embedding como numpy array"""
        if self.text_embedding:
            return np.array(self.text_embedding)
        return None
    
    class Config:
        arbitrary_types_allowed = True

class QueryRequirement(BaseModel):
    """Requerimiento del usuario"""
    feature_name: str = Field(..., description="Feature buscado: cocina_equipada")
    importance: float = Field(1.0, ge=0, le=1, description="Importancia 0-1")
    required: bool = Field(False, description="Si es obligatorio")

class MatchResult(BaseModel):
    """Resultado de matching property vs query"""
    property_id: str
    
    # Scores individuales
    feature_match_score: float = Field(0.0, ge=0, le=1)
    semantic_similarity_score: float = Field(0.0, ge=0, le=1)
    
    # Score final combinado
    final_score: float = Field(0.0, ge=0, le=1)
    
    # Detalles
    matched_features: List[DetectedFeature] = Field(default_factory=list)
    missing_requirements: List[str] = Field(default_factory=list)
    
    def is_good_match(self, threshold: float = 0.7) -> bool:
        """Es un buen match?"""
        return self.final_score >= threshold