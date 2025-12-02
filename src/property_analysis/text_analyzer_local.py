"""
Text Analyzer v2.0 - Flexible con embeddings
"""
import json
import os
from typing import Optional, List
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from anthropic import Anthropic
from dotenv import load_dotenv
import mlflow

load_dotenv()

from .schemas import (
    PropertyAnalysis, DetectedFeature, QueryRequirement, MatchResult
)
from .prompts import get_feature_extraction_prompt

class PropertyTextAnalyzer:
    """
    Analyzer flexible con:
    - Feature extraction dinámica
    - Embeddings semánticos
    - Matching inteligente
    """
    
    def __init__(
        self,
        llm_model: str = "claude-sonnet-4-20250514",
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
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
        
        # Embedding model (multilingual para español)
        print(f"Loading embedding model: {embedding_model}...")
        self.embedding_model = SentenceTransformer(embedding_model)
        print("✅ Embedding model loaded")
        
        # Cache
        self.cache_dir = Path(cache_dir or "data/cache/embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze(
        self,
        property_id: str,
        description: str,
        generate_embedding: bool = True,
        log_to_mlflow: bool = False
    ) -> PropertyAnalysis:
        """
        Analiza propiedad con feature extraction + embeddings
        
        Args:
            property_id: ID único
            description: Descripción textual
            generate_embedding: Si True, genera embedding (más lento)
            log_to_mlflow: Log a MLflow
            
        Returns:
            PropertyAnalysis completo
        """
        if not description or len(description.strip()) < 20:
            return PropertyAnalysis(
                property_id=property_id,
                description_length=len(description)
            )
        
        # 1. Feature extraction con LLM
        features = self._extract_features_with_llm(description)
        
        # 2. Generate embedding
        embedding = None
        if generate_embedding:
            embedding = self._generate_embedding(property_id, description)
        
        # 3. Quality score (promedio de top features)
        top_features = sorted(features, key=lambda f: f.confidence, reverse=True)[:5]
        quality_score = np.mean([f.confidence for f in top_features]) if top_features else 0.0
        
        result = PropertyAnalysis(
            property_id=property_id,
            detected_features=features,
            text_embedding=embedding.tolist() if embedding is not None else None,
            description_length=len(description),
            overall_quality_score=float(quality_score)
        )
        
        # MLflow logging
        if log_to_mlflow and mlflow.active_run():
            mlflow.log_param(f"{property_id}_prompt_version", self.prompt_version)
            mlflow.log_metric(f"{property_id}_num_features", len(features))
            mlflow.log_metric(f"{property_id}_quality_score", quality_score)
        
        return result
    
    def _extract_features_with_llm(self, description: str) -> List[DetectedFeature]:
        """Extrae features usando LLM"""
        try:
            # Truncar si es muy largo
            desc_truncated = description[:3000]
            
            response = self.llm_client.messages.create(
                model=self.llm_model,
                max_tokens=2000,
                system=self.prompt_config["system"],
                messages=[{"role": "user", "content": desc_truncated}]
            )
            
            response_text = response.content[0].text.strip()
            
            # Limpiar markdown
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            response_text = response_text.strip()
            
            # Parse
            result = json.loads(response_text)
            features = [DetectedFeature(**f) for f in result.get("detected_features", [])]
            
            return features
            
        except Exception as e:
            print(f"⚠️  Feature extraction error: {e}")
            return []
    
    def _generate_embedding(self, property_id: str, text: str) -> Optional[np.ndarray]:
        """
        Genera embedding con cache
        
        Returns:
            numpy array de 768 dims
        """
        # Check cache
        cache_file = self.cache_dir / f"{property_id}.npy"
        if cache_file.exists():
            return np.load(cache_file)
        
        # Generate
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        
        # Save to cache
        np.save(cache_file, embedding)
        
        return embedding
    
    def match_against_query(
        self,
        property_analysis: PropertyAnalysis,
        query_text: str,
        required_features: Optional[List[QueryRequirement]] = None,
        feature_weight: float = 0.6,
        semantic_weight: float = 0.4
    ) -> MatchResult:
        """
        Match propiedad contra query del usuario
        
        Args:
            property_analysis: Análisis de la propiedad
            query_text: Query en lenguaje natural
            required_features: Features requeridos (opcional)
            feature_weight: Peso del feature matching (0-1)
            semantic_weight: Peso de similitud semántica (0-1)
            
        Returns:
            MatchResult con scores
        """
        # 1. Feature matching
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
                    # Fuzzy match (sinónimos)
                    fuzzy_match = self._fuzzy_feature_match(
                        req.feature_name,
                        property_analysis.detected_features
                    )
                    if fuzzy_match:
                        matched.append(fuzzy_match)
                        feature_score += fuzzy_match.confidence * req.importance * 0.8
                    else:
                        missing.append(req.feature_name)
            
            feature_score = feature_score / sum(r.importance for r in required_features)
        
        # 2. Semantic similarity
        semantic_score = 0.0
        if property_analysis.text_embedding:
            query_embedding = self.embedding_model.encode(query_text, convert_to_numpy=True)
            property_embedding = property_analysis.get_embedding_array()
            
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                property_embedding.reshape(1, -1)
            )[0][0]
            semantic_score = float(similarity)
        
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
    
    def _fuzzy_feature_match(
        self,
        target: str,
        features: List[DetectedFeature],
        threshold: float = 0.7
    ) -> Optional[DetectedFeature]:
        """Busca match fuzzy (sinónimos)"""
        target_embedding = self.embedding_model.encode(target, convert_to_numpy=True)
        
        best_match = None
        best_similarity = 0.0
        
        for feature in features:
            feature_embedding = self.embedding_model.encode(feature.name, convert_to_numpy=True)
            similarity = cosine_similarity(
                target_embedding.reshape(1, -1),
                feature_embedding.reshape(1, -1)
            )[0][0]
            
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
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