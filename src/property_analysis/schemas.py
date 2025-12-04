"""
Schemas flexibles para análisis de propiedades
UPDATED: Added CombinedAnalysis for multi-modal analysis
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
    evidence: Optional[str] = Field(None, description="Evidencia que justifica la detección")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "terraza_privada",
                "value": "20m²",
                "confidence": 0.95,
                "source": "description",
                "evidence": "Mencionado explícitamente en la descripción"
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


class VisionAnalysis(BaseModel):
    """
    Resultados del análisis visual
    NEW: Added for Phase 3A
    """
    property_id: str
    
    # Features detectados de imágenes (Stage 2: Claude Vision)
    detected_features: List[DetectedFeature] = Field(default_factory=list)
    
    # Image embeddings (Stage 1: CLIP)
    # Stored as list for serialization, use get_embeddings_array() for numpy
    image_embeddings: Optional[List[List[float]]] = Field(None, exclude=True)
    
    # Metadata
    analyzed_images: List[str] = Field(default_factory=list)
    analysis_stage: str = Field("stage1", description="stage1 (CLIP) or stage2 (Claude)")
    total_cost_eur: float = Field(0.0, description="Cost of Claude Vision analysis")
    
    def get_embeddings_array(self) -> Optional[List[np.ndarray]]:
        """Convert embeddings to numpy arrays"""
        if self.image_embeddings:
            return [np.array(emb) for emb in self.image_embeddings]
        return None
    
    def get_avg_embedding(self) -> Optional[np.ndarray]:
        """Average embedding across all images"""
        embeddings = self.get_embeddings_array()
        if embeddings:
            return np.mean(embeddings, axis=0)
        return None
    
    def get_feature(self, name: str) -> Optional[DetectedFeature]:
        """Get feature by name"""
        for f in self.detected_features:
            if f.name.lower() == name.lower():
                return f
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


class CombinedAnalysis(BaseModel):
    """
    Resultado completo del análisis multi-modal
    NEW: Added for Phase 3A
    
    Combina análisis de texto + visión para ranking final
    """
    property_id: str
    
    # Análisis individuales
    text_analysis: PropertyAnalysis
    vision_analysis: Optional[VisionAnalysis] = None
    
    # Scoring
    match_result: MatchResult
    
    # Metadata
    analysis_stage: str = Field("stage1", description="stage1 (CLIP) or stage2 (Claude)")
    processing_time_seconds: Optional[float] = None
    
    @property
    def final_score(self) -> float:
        """Shortcut to final score"""
        return self.match_result.final_score
    
    @property
    def all_features(self) -> List[DetectedFeature]:
        """All features from text + vision"""
        features = self.text_analysis.detected_features.copy()
        if self.vision_analysis:
            features.extend(self.vision_analysis.detected_features)
        return features
    
    def get_feature_sources(self) -> Dict[str, List[str]]:
        """Get sources for each detected feature"""
        sources = {}
        
        for f in self.text_analysis.detected_features:
            sources.setdefault(f.name, []).append('text')
        
        if self.vision_analysis:
            for f in self.vision_analysis.detected_features:
                sources.setdefault(f.name, []).append('vision')
        
        return sources
    
    def explain_score(self) -> str:
        """Human-readable score explanation"""
        lines = [
            f"Property: {self.property_id}",
            f"Final Score: {self.final_score:.3f}",
            f"",
            f"Component Scores:",
            f"  • Features: {self.match_result.feature_match_score:.3f}",
            f"  • Semantic: {self.match_result.semantic_similarity_score:.3f}",
            f"",
            f"Matched Features ({len(self.match_result.matched_features)}):"
        ]
        
        for f in self.match_result.matched_features:
            sources = self.get_feature_sources().get(f.name, [])
            source_str = "+".join(sources)
            lines.append(f"  • {f.name}: {f.confidence:.2f} [{source_str}]")
        
        if self.match_result.missing_requirements:
            lines.append(f"")
            lines.append(f"Missing ({len(self.match_result.missing_requirements)}):")
            for m in self.match_result.missing_requirements:
                lines.append(f"  • {m}")
        
        return "\n".join(lines)
    
    class Config:
        arbitrary_types_allowed = True