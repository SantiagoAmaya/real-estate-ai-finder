"""
Flexible Text Analyzer con múltiples backends
- API: Claude API (rápido, para desarrollo)
- Local: sentence-transformers (lento, para producción)
- NUEVO: Tool Calling para Structured Outputs (sin errores JSON)
"""
import json
import os
import re
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

# Tool Calling schemas con fallback inline
try:
    from .schemas import (
        PropertyFeaturesResponse, FeatureSchema,
        feature_schema_to_detected_feature
    )
except ImportError:
    # Si schemas.py no tiene el helper, usar fallback
    from .schemas import PropertyFeaturesResponse, FeatureSchema
    
    def feature_schema_to_detected_feature(fs) -> DetectedFeature:
        return DetectedFeature(
            name=fs.name,
            value=fs.value,
            confidence=fs.confidence,
            source=fs.source,
            evidence=getattr(fs, 'evidence', None)
        )

from .prompts import get_feature_extraction_prompt

class PropertyTextAnalyzer:
    """
    Analyzer flexible con backend intercambiable
    
    Backends:
    - 'api': Claude API para embeddings (rápido, ~1-2s por propiedad)
    - 'local': sentence-transformers (lento primera vez, luego rápido)
    
    NUEVO: Usa Tool Calling para feature extraction (100% JSON válido)
    """
    
    def __init__(
        self,
        backend: Literal["api", "local"] = "api",
        llm_model: str = "claude-sonnet-4-20250514",
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        prompt_version: str = "v2.0",
        cache_dir: Optional[str] = None,
        api_key: Optional[str] = None,
        use_tool_calling: bool = True  # NUEVO: activar/desactivar Tool Calling
    ):
        self.backend = backend
        self.llm_model = llm_model
        self.embedding_model_name = embedding_model
        self.prompt_version = prompt_version
        self.prompt_config = get_feature_extraction_prompt(prompt_version)
        self.use_tool_calling = use_tool_calling
        
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
        
        mode = "Tool Calling" if use_tool_calling else "JSON text"
        print(f"✅ Analyzer ready (backend: {backend}, extraction: {mode})")
    
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
        
        # 1. Feature extraction con LLM (Tool Calling o legacy)
        if self.use_tool_calling:
            features = self._extract_features_tool_calling(description)
        else:
            features = self._extract_features_legacy(description)
        
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
            overall_quality_score=float(quality_score),
            original_text=description  # Guardar para verbose mode
        )
        
        # MLflow logging
        if log_to_mlflow and mlflow.active_run():
            mlflow.log_param(f"{property_id}_backend", self.backend)
            mlflow.log_param(f"{property_id}_prompt_version", self.prompt_version)
            mlflow.log_param(f"{property_id}_extraction_mode", "tool_calling" if self.use_tool_calling else "legacy")
            mlflow.log_metric(f"{property_id}_num_features", len(features))
            mlflow.log_metric(f"{property_id}_quality_score", quality_score)
        
        return result
    
    def _extract_features_tool_calling(self, description: str) -> List[DetectedFeature]:
        """
        Extrae features usando Anthropic Tool Calling (100% JSON válido)
        
        Tool Calling garantiza que la respuesta siempre sea JSON válido
        que cumple con el schema de Pydantic. No más errores de parsing!
        """
        try:
            desc_truncated = description[:3000]
            
            # Convertir Pydantic schema a JSON Schema para Anthropic
            tool_schema = {
                "name": "extract_property_features",
                "description": "Extrae características estructuradas de una descripción de propiedad inmobiliaria",
                "input_schema": PropertyFeaturesResponse.model_json_schema()
            }
            
            response = self.llm_client.messages.create(
                model=self.llm_model,
                max_tokens=2000,
                tools=[tool_schema],
                messages=[{
                    "role": "user",
                    "content": f"{self.prompt_config['system']}\n\nDESCRIPCIÓN:\n{desc_truncated}"
                }]
            )
            
            # Extract tool use from response
            for block in response.content:
                if block.type == "tool_use" and block.name == "extract_property_features":
                    # Pydantic valida automáticamente
                    validated = PropertyFeaturesResponse(**block.input)
                    
                    # Convert to DetectedFeature objects usando helper
                    features = [
                        feature_schema_to_detected_feature(f)
                        for f in validated.detected_features
                    ]
                    return features
            
            # Fallback: si no hay tool use, intentar parsing legacy
            print("⚠️  No tool use in response, trying legacy parsing...")
            for block in response.content:
                if hasattr(block, 'text'):
                    return self._extract_features_legacy_parse(block.text)
            
            return []
            
        except Exception as e:
            print(f"⚠️  Tool calling error: {e}")
            # Fallback completo a legacy
            return self._extract_features_legacy(description)
    
    def _extract_features_legacy(self, description: str) -> List[DetectedFeature]:
        """Método legacy con retry para JSON malformado"""
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                desc_truncated = description[:3000]
                
                response = self.llm_client.messages.create(
                    model=self.llm_model,
                    max_tokens=2000,
                    system=self.prompt_config["system"],
                    messages=[{"role": "user", "content": desc_truncated}]
                )
                
                response_text = response.content[0].text.strip()
                return self._extract_features_legacy_parse(response_text)
                
            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    print(f"⚠️  JSON parse error (attempt {attempt+1}/{max_retries}), retrying...")
                    continue
                else:
                    print(f"⚠️  Feature extraction error: {e}")
                    return []
            except Exception as e:
                print(f"⚠️  Feature extraction error: {e}")
                return []
        
        return []
    
    def _extract_features_legacy_parse(self, response_text: str) -> List[DetectedFeature]:
        """Parse JSON legacy con limpieza robusta"""
        # Limpiar markdown
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        response_text = response_text.strip()
        
        # ROBUST JSON PARSING
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to fix common issues
            # Remove trailing commas
            response_text = re.sub(r',(\s*[}\]])', r'\1', response_text)
            
            # Fix missing commas between objects
            response_text = re.sub(r'}\s*{', '},{', response_text)
            
            # Remove unescaped newlines in strings
            response_text = re.sub(r'(?<!\\)\n', ' ', response_text)
            
            # Try again
            result = json.loads(response_text)
        
        features = [DetectedFeature(**f) for f in result.get("detected_features", [])]
        return features
    
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