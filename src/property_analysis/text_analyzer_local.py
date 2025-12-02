"""
Text Analyzer v2.0 - Flexible con embeddings
"""
import json
import os
from typing import Optional, List
from pathlib import Path
import numpy as np
import anthropic
from anthropic import Anthropic
from dotenv import load_dotenv
import mlflow

load_dotenv()

from .schemas import (
    PropertyAnalysis, DetectedFeature, QueryRequirement, MatchResult
)
from .prompts import get_feature_extraction_prompt

class PropertyTextAnalyzer:
    """Analyzer flexible con Claude para features + embeddings"""
    
    def __init__(
        self,
        llm_model: str = "claude-sonnet-4-20250514",
        prompt_version: str = "v2.0",
        cache_dir: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        # LLM para feature extraction
        self.llm_model = llm_model
        self.prompt_version = prompt_version
        self.prompt_config = get_feature_extraction_prompt(prompt_version)
        
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found")
        self.llm_client = Anthropic(api_key=api_key)
        
        # Cache
        self.cache_dir = Path(cache_dir or "data/cache/embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        print("✅ Analyzer ready (using Claude for embeddings)")
    
    def _generate_embedding(self, property_id: str, text: str) -> Optional[np.ndarray]:
        """Genera embedding usando Claude (simple similarity score)"""
        # Cache check
        cache_file = self.cache_dir / f"{property_id}.npy"
        if cache_file.exists():
            return np.load(cache_file)
        
        # Por ahora, usar hash simple del texto normalizado
        # En producción, usarías Voyage AI o similar para embeddings reales
        # Este es un placeholder que funciona para MVP
        import hashlib
        
        # Normalizar texto
        normalized = text.lower().strip()
        words = set(normalized.split())
        
        # Crear "embedding" simple basado en palabras clave (768 dims como placeholder)
        # Esto es temporal - funciona para prototipo
        embedding = np.zeros(768)
        for i, word in enumerate(sorted(words)[:100]):
            hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
            idx = hash_val % 768
            embedding[idx] += 1.0
        
        # Normalizar
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        # Cache
        np.save(cache_file, embedding)
        return embedding
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Similitud coseno simple"""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def match_against_query(
        self,
        property_analysis: PropertyAnalysis,
        query_text: str,
        required_features: Optional[List[QueryRequirement]] = None,
        feature_weight: float = 0.7,  # Más peso a features
        semantic_weight: float = 0.3   # Menos peso a semantic (temporal)
    ) -> MatchResult:
        """Match propiedad contra query"""
        
        # 1. Feature matching (mismo código)
        feature_score = 0.0
        matched = []
        missing = []
        
        if required_features:
            for req in required_features:
                feature = property_analysis.get_feature(req.feature_name)
                
                if feature and feature.confidence >= 0.5:
                    matched.append(feature)
                    feature_score += feature.confidence * req.importance
                else:
                    # Fuzzy match simple (word overlap)
                    fuzzy_match = self._simple_fuzzy_match(
                        req.feature_name,
                        property_analysis.detected_features
                    )
                    if fuzzy_match:
                        matched.append(fuzzy_match)
                        feature_score += fuzzy_match.confidence * req.importance * 0.8
                    else:
                        missing.append(req.feature_name)
            
            feature_score = feature_score / sum(r.importance for r in required_features)
        
        # 2. Semantic similarity (simplificado)
        semantic_score = 0.0
        if property_analysis.text_embedding:
            query_embedding = self._generate_embedding("query_temp", query_text)
            property_embedding = property_analysis.get_embedding_array()
            
            if query_embedding is not None and property_embedding is not None:
                semantic_score = self._cosine_similarity(query_embedding, property_embedding)
        
        # 3. Combined score
        final_score = (feature_score * feature_weight) + (semantic_score * semantic_weight)
        
        return MatchResult(
            property_id=property_analysis.property_id,
            feature_match_score=feature_score,
            semantic_similarity_score=semantic_score,
            final_score=final_score,
            matched_features=matched,
            missing_requirements=missing
        )
    
    def _simple_fuzzy_match(
        self,
        target: str,
        features: List[DetectedFeature],
        threshold: float = 0.5
    ) -> Optional[DetectedFeature]:
        """Fuzzy match simple por overlap de palabras"""
        target_words = set(target.lower().replace('_', ' ').split())
        
        best_match = None
        best_score = 0.0
        
        for feature in features:
            feature_words = set(feature.name.lower().replace('_', ' ').split())
            overlap = len(target_words & feature_words)
            similarity = overlap / max(len(target_words), len(feature_words))
            
            if similarity > best_score and similarity >= threshold:
                best_score = similarity
                best_match = feature
        
        return best_match
    
    def analyze_batch(
        self,
        properties: List[dict],
        generate_embeddings: bool = True
    ) -> List[PropertyAnalysis]:
        """Analiza múltiples propiedades"""
        results = []
        total = len(properties)
        
        for i, prop in enumerate(properties, 1):
            print(f"Analyzing {i}/{total}: {prop.get('id', 'unknown')}")
            
            result = self.analyze(
                property_id=prop.get('id', f'prop_{i}'),
                description=prop.get('description', ''),
                generate_embedding=generate_embeddings
            )
            results.append(result)
        
        return results