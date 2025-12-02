"""
Flexible Text Analyzer con múltiples backends
- API: Claude API (rápido, para desarrollo)
- Local: sentence-transformers (lento, para producción)
"""
import json
import os
from typing import Optional, List, Literal
from pathlib import Path
import numpy as np
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
    Analyzer flexible con backend intercambiable
    
    Backends:
    - 'api': Claude API para embeddings (rápido, ~1-2s por propiedad)
    - 'local': sentence-transformers (lento primera vez, luego rápido)
    """
    
    def __init__(
        self,
        backend: Literal["api", "local"] = "api",
        llm_model: str = "claude-sonnet-4-20250514",
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        prompt_version: str = "v2.0",
        cache_dir: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        self.backend = backend
        self.llm_model = llm_model
        self.embedding_model_name = embedding_model
        self.prompt_version = prompt_version
        self.prompt_config = get_feature_extraction_prompt(prompt_version)
        
        # API key para Claude
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found")
        self.llm_client = Anthropic(api_key=api_key)
        
        # Cache
        self.cache_dir = Path(cache_dir or "data/cache/embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Lazy load embedding model (solo si se usa backend local)
        self._local_embedding_model = None
        
        print(f"✅ Analyzer ready (backend: {backend})")
    
    @property
    def local_embedding_model(self):
        """Lazy load del modelo local"""
        if self._local_embedding_model is None:
            if self.backend == "local":
                print(f"Loading local embedding model: {self.embedding_model_name}...")
                from sentence_transformers import SentenceTransformer
                self._local_embedding_model = SentenceTransformer(self.embedding_model_name)
                print("✅ Local model loaded")
        return self._local_embedding_model
    
    def analyze(
        self,
        property_id: str,
        description: str,
        generate_embedding: bool = True,
        log_to_mlflow: bool = False
    ) -> PropertyAnalysis:
        """Analiza propiedad con features + embeddings"""
        
        if not description or len(description.strip()) < 20:
            return PropertyAnalysis(
                property_id=property_id,
                description_length=len(description)
            )
        
        # 1. Feature extraction con LLM
        features = self._extract_features_with_llm(description)
        
        # 2. Generate embedding según backend
        embedding = None
        if generate_embedding:
            if self.backend == "api":
                embedding = self._generate_embedding_api(property_id, description)
            else:  # local
                embedding = self._generate_embedding_local(property_id, description)
        
        # 3. Quality score
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
            mlflow.log_param(f"{property_id}_backend", self.backend)
            mlflow.log_param(f"{property_id}_prompt_version", self.prompt_version)
            mlflow.log_metric(f"{property_id}_num_features", len(features))
            mlflow.log_metric(f"{property_id}_quality_score", quality_score)
        
        return result
    
    def _extract_features_with_llm(self, description: str) -> List[DetectedFeature]:
        """Extrae features usando LLM"""
        try:
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
            
            result = json.loads(response_text)
            features = [DetectedFeature(**f) for f in result.get("detected_features", [])]
            
            return features
            
        except Exception as e:
            print(f"⚠️  Feature extraction error: {e}")
            return []
    
    def _generate_embedding_api(self, property_id: str, text: str) -> Optional[np.ndarray]:
        """
        Genera embedding usando Claude API (aproximación simple)
        
        NOTA: Esto es un placeholder. Para producción real, usa:
        - Voyage AI embeddings API
        - OpenAI embeddings API
        - O mantén local para control total
        """
        cache_file = self.cache_dir / f"{property_id}_api.npy"
        if cache_file.exists():
            return np.load(cache_file)
        
        # Embedding simple basado en TF-IDF + normalización
        # Suficiente para MVP, no para producción final
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Crear vectorizer con dims fijas (768)
        vectorizer = TfidfVectorizer(max_features=768, ngram_range=(1, 2))
        
        # Necesitamos al menos 2 documentos para fit
        # Usamos el texto + una versión limpia
        corpus = [text, text.lower().strip()]
        
        try:
            tfidf_matrix = vectorizer.fit_transform(corpus)
            embedding = tfidf_matrix[0].toarray().flatten()
            
            # Pad/truncate a 768
            if len(embedding) < 768:
                embedding = np.pad(embedding, (0, 768 - len(embedding)))
            else:
                embedding = embedding[:768]
            
            # Normalizar
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            # Cache
            np.save(cache_file, embedding)
            return embedding
            
        except Exception as e:
            print(f"⚠️  API embedding error: {e}")
            return None
    
    def _generate_embedding_local(self, property_id: str, text: str) -> Optional[np.ndarray]:
        """Genera embedding con sentence-transformers (preciso pero lento)"""
        cache_file = self.cache_dir / f"{property_id}_local.npy"
        if cache_file.exists():
            return np.load(cache_file)
        
        try:
            embedding = self.local_embedding_model.encode(text, convert_to_numpy=True)
            np.save(cache_file, embedding)
            return embedding
        except Exception as e:
            print(f"⚠️  Local embedding error: {e}")
            return None
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Similitud coseno"""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    
    def match_against_query(
        self,
        property_analysis: PropertyAnalysis,
        query_text: str,
        required_features: Optional[List[QueryRequirement]] = None,
        feature_weight: float = 0.7,
        semantic_weight: float = 0.3
    ) -> MatchResult:
        """Match propiedad contra query"""
        
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
        
        # 2. Semantic similarity
        semantic_score = 0.0
        if property_analysis.text_embedding:
            if self.backend == "api":
                query_embedding = self._generate_embedding_api("query_temp", query_text)
            else:
                query_embedding = self._generate_embedding_local("query_temp", query_text)
            
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