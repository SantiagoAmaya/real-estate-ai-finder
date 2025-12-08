"""
Vision Analyzer with conditional imports for Railway compatibility

Supports:
- Claude Vision API (always available)
- Qwen2-VL local (optional, requires GPU + torch)

For Railway: Only Claude modes work (qwen requires GPU)
"""
from typing import List, Optional, Dict, Tuple
from pathlib import Path
import base64
import requests
from io import BytesIO
from PIL import Image
from anthropic import Anthropic
import os

from .schemas import VisionAnalysis, DetectedFeature


class VisionAnalyzer:
    """
    Multi-backend vision analyzer with Railway compatibility
    
    Modes:
    - claude_only: Claude Vision API only ✅ Railway
    - claude_primary: Claude with Qwen fallback ✅ Railway (if no GPU, stays Claude)
    - qwen_only: Local Qwen only ❌ Railway (requires GPU)
    - qwen_primary: Qwen with Claude fallback ❌ Railway (requires GPU)
    """
    
    def __init__(
        self,
        mode: str = "claude_primary",
        qwen_confidence_threshold: float = 0.7,
        max_claude_calls: int = 100,
        cache_dir: Optional[str] = None
    ):
        self.mode = mode
        self.qwen_confidence_threshold = qwen_confidence_threshold
        self.max_claude_calls = max_claude_calls
        self.cache_dir = Path(cache_dir or "data/cache/vision")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Claude
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found")
        
        self.claude_client = Anthropic(api_key=api_key)
        self.claude_model = "claude-sonnet-4-20250514"
        
        # Budget tracking
        self.claude_calls_made = 0
        self.total_cost = 0.0
        
        # Try to load Qwen (optional)
        self._qwen_analyzer = None
        self._qwen_available = False
        
        if mode in ["qwen_only", "qwen_primary"]:
            self._try_load_qwen()
        
        print(f"✅ VisionAnalyzer initialized")
        print(f"   Mode: {self.mode}")
        print(f"   Qwen available: {self._qwen_available}")
    
    def _try_load_qwen(self):
        """Try to load Qwen (optional, graceful failure)"""
        try:
            # Try importing Qwen
            from .local_vision_qwen import Qwen2VisionAnalyzer
            
            self._qwen_analyzer = Qwen2VisionAnalyzer(
                cache_dir=str(self.cache_dir / "qwen")
            )
            self._qwen_available = True
            print("✅ Qwen2-VL loaded successfully")
            
        except ImportError as e:
            print(f"⚠️  Qwen not available: {e}")
            print(f"⚠️  This is expected on Railway (no GPU)")
            
            # If qwen_only was explicitly requested, error
            if self.mode == "qwen_only":
                raise RuntimeError(
                    "qwen_only mode requested but Qwen not available. "
                    "For Railway, use 'claude_only' or 'claude_primary'."
                )
            
            # Otherwise, auto-fallback to Claude
            if self.mode == "qwen_primary":
                print(f"⚠️  Falling back to claude_primary mode")
                self.mode = "claude_primary"
            
            self._qwen_available = False
        
        except Exception as e:
            print(f"⚠️  Error loading Qwen: {e}")
            self._qwen_available = False
    
    def generate_photo_description(
        self,
        image_urls: List[str],
        max_images: int = 3,
        user_query: str = ""
    ) -> str:
        """
        Generate consolidated description from property photos
        
        Routes to Claude or Qwen based on mode and availability
        """
        if not image_urls:
            return ""
        
        image_urls = image_urls[:max_images]
        
        # Route based on mode
        if self.mode == "qwen_only":
            if not self._qwen_available:
                raise RuntimeError("qwen_only mode but Qwen not available")
            return self._analyze_with_qwen(image_urls, user_query)
        
        elif self.mode == "claude_only":
            return self._analyze_with_claude(image_urls, user_query)
        
        elif self.mode == "claude_primary":
            # Claude first (always available)
            return self._analyze_with_claude(image_urls, user_query)
        
        elif self.mode == "qwen_primary":
            # Try Qwen first if available
            if self._qwen_available:
                try:
                    return self._analyze_with_qwen(image_urls, user_query)
                except Exception as e:
                    print(f"⚠️  Qwen failed: {e}, falling back to Claude")
            
            # Fallback to Claude
            return self._analyze_with_claude(image_urls, user_query)
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _analyze_with_claude(
        self,
        image_urls: List[str],
        user_query: str
    ) -> str:
        """Analyze with Claude Vision API"""
        
        # Check budget
        if self.claude_calls_made >= self.max_claude_calls:
            print(f"⚠️  Claude budget exhausted")
            return ""
        
        individual_descriptions = []
        
        for idx, url in enumerate(image_urls, 1):
            try:
                # Download image
                image = self._download_image(url)
                if image is None:
                    continue
                
                # Convert to base64
                image_base64 = self._image_to_base64(image)
                
                # Query-aware prompt
                prompt = self._get_vision_prompt(user_query)
                
                # Call Claude
                response = self.claude_client.messages.create(
                    model=self.claude_model,
                    max_tokens=400,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }]
                )
                
                description = response.content[0].text.strip()
                individual_descriptions.append(description)
                
                # Update budget
                self.claude_calls_made += 1
                self.total_cost += 0.025
                
                print(f"  ✅ Image {idx}/{len(image_urls)} (Claude)")
                
            except Exception as e:
                print(f"  ⚠️  Error image {idx}: {e}")
                continue
        
        if not individual_descriptions:
            return ""
        
        if len(individual_descriptions) == 1:
            return individual_descriptions[0]
        
        # Consolidate multiple descriptions
        return self._consolidate_descriptions(individual_descriptions, user_query)
    
    def _analyze_with_qwen(
        self,
        image_urls: List[str],
        user_query: str
    ) -> str:
        """Analyze with Qwen (local GPU)"""
        if not self._qwen_available:
            raise RuntimeError("Qwen not available")
        
        prompt = self._get_vision_prompt(user_query)
        description, confidence = self._qwen_analyzer.generate_description(
            image_urls=image_urls,
            prompt_template=prompt,
            max_images=len(image_urls)
        )
        
        return description
    
    def _get_vision_prompt(self, user_query: str) -> str:
        """Get query-aware vision prompt"""
        base_prompt = """Analiza esta foto de forma OBJETIVA y FACTUAL, sin embellecer.

Describe SOLO lo que ves:
1. Tipo de espacio: Local comercial, piso, oficina
2. Acceso: Entrada desde calle (independiente) o portal compartido
3. Techos: Altura en metros (2.5m estándar, 3.0m medio-alto, 3.5m+ alto)
4. Suelos: Material exacto (cerámica, parquet, hormigón)
5. Luz natural: Número de ventanas/puertas visibles
6. Distribución: Diáfano, compartimentado
7. Estado: Nuevo/reformado/antiguo

NO uses adjetivos comerciales. Máximo 150 palabras."""
        
        if user_query:
            base_prompt += f"\n\nCONTEXTO: El usuario busca '{user_query}'. Prioriza información relevante."
        
        return base_prompt
    
    def _consolidate_descriptions(
        self,
        individual_descriptions: List[str],
        user_query: str
    ) -> str:
        """Consolidate multiple descriptions"""
        # Don't label images, just list them
        combined = "\n---\n".join(individual_descriptions)
        
        prompt = f"""Tienes {len(individual_descriptions)} descripciones de IMÁGENES DE LA MISMA PROPIEDAD.

DESCRIPCIONES (de diferentes ángulos/espacios de la misma propiedad):
{combined}

BÚSQUEDA DEL USUARIO: "{user_query}"

TAREA: Sintetiza en UN SOLO PÁRRAFO FLUIDO (máximo 150 palabras) que:
- Unifique la información de todos los espacios mostrados
- Elimine redundancias (no repitas "techos 2.5m" 3 veces)
- Priorice lo relevante para la búsqueda
- Sea factual y objetivo
- Incluya: tipo, acceso, techos, suelos, luz, distribución, estado

IMPORTANTE: 
- NO uses "Imagen 1", "Imagen 2" en tu respuesta
- Escribe un párrafo corrido y natural

Responde SOLO con el párrafo en español."""
        
        try:
            response = self.claude_client.messages.create(
                model=self.claude_model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            
            consolidated = response.content[0].text.strip()
            
            # Clean up markdown
            if consolidated.startswith("```"):
                consolidated = consolidated.split("```")[1].strip()
                if consolidated.startswith("json") or consolidated.startswith("text"):
                    consolidated = consolidated[4:].strip()
            
            if consolidated.startswith('"') and consolidated.endswith('"'):
                consolidated = consolidated[1:-1]
            
            self.claude_calls_made += 1
            self.total_cost += 0.003
            
            return consolidated
            
        except Exception as e:
            print(f"⚠️  Consolidation error: {e}")
            return "\n\n".join(individual_descriptions[:2])
    
    def _download_image(self, url: str) -> Optional[Image.Image]:
        """Download image from URL"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            print(f"⚠️  Failed to download {url}: {e}")
            return None
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64"""
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode()
    
    def analyze_property_stage1(
        self,
        property_id: str,
        image_urls: List[str],
        max_images: int = 3
    ) -> VisionAnalysis:
        """
        Stage 1: CLIP embeddings only (if Qwen available)
        
        Returns empty VisionAnalysis if Qwen not available
        """
        if not self._qwen_available:
            # Return empty analysis
            return VisionAnalysis(
                property_id=property_id,
                image_embeddings={},
                avg_embedding=None,
                detected_features=[],
                confidence_score=0.0
            )
        
        # Use Qwen CLIP
        embeddings = {}
        for i, url in enumerate(image_urls[:max_images]):
            embedding = self._qwen_analyzer.generate_clip_embedding(url)
            if embedding is not None:
                embeddings[f"image_{i}"] = embedding
        
        return VisionAnalysis(
            property_id=property_id,
            image_embeddings=embeddings,
            avg_embedding=None,  # Computed on demand
            detected_features=[],
            confidence_score=1.0
        )
    
    def analyze_property_stage2(
        self,
        property_id: str,
        image_urls: List[str],
        query_text: str,
        target_features: List[str],
        max_images: int = 2
    ) -> VisionAnalysis:
        """
        Stage 2: Claude Vision for detailed features
        COSTS MONEY
        """
        # Generate description
        description = self.generate_photo_description(
            image_urls=image_urls,
            max_images=max_images,
            user_query=query_text
        )
        
        # Dummy detected features (text_analyzer will extract from description)
        return VisionAnalysis(
            property_id=property_id,
            image_embeddings={},
            avg_embedding=None,
            detected_features=[],  # Extracted by text_analyzer from description
            confidence_score=0.9
        )
    
    def compute_image_text_similarity(
        self,
        image_embedding,
        query_text: str
    ) -> float:
        """Compute similarity (requires Qwen CLIP)"""
        if not self._qwen_available:
            return 0.0
        
        # Use Qwen CLIP
        text_embedding = self._qwen_analyzer.generate_text_embedding(query_text)
        
        import numpy as np
        similarity = np.dot(image_embedding, text_embedding) / (
            np.linalg.norm(image_embedding) * np.linalg.norm(text_embedding)
        )
        return float(similarity)
    
    def get_budget_status(self) -> Dict:
        """Get budget status"""
        # Get qwen stats if available
        qwen_calls = 0
        if self._qwen_available and self._qwen_analyzer:
            qwen_calls = getattr(self._qwen_analyzer, 'calls_made', 0)
        
        return {
            'mode': self.mode,
            'qwen_available': self._qwen_available,
            'claude_calls_made': self.claude_calls_made,
            'qwen_calls_made': qwen_calls,  # For backward compatibility
            'max_claude_calls': self.max_claude_calls,
            'remaining': self.max_claude_calls - self.claude_calls_made,
            'total_cost_eur': self.total_cost
        }