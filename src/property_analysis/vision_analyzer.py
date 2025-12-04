"""
Hybrid Vision Analyzer: CLIP (free) + Claude Vision (selective)

Architecture:
  Stage 1: CLIP embeddings for ALL properties (M3 GPU, free)
           → Quick image-text similarity scoring
  
  Stage 2: Claude Vision for TOP candidates only (budget-conscious)
           → Dynamic feature extraction (like text analyzer)

Budget: €20 = ~300 Claude Vision calls (2 images each)
"""
import os
from typing import List, Optional, Literal, Tuple
from pathlib import Path
import numpy as np
from anthropic import Anthropic
from dotenv import load_dotenv
import json
import mlflow
from PIL import Image
import requests
from io import BytesIO

load_dotenv()

from .schemas import DetectedFeature, QueryRequirement
from .local_vision_qwen import Qwen2VisionAnalyzer

class VisionAnalysis:
    """Results from vision analysis"""
    def __init__(
        self,
        property_id: str,
        detected_features: List[DetectedFeature] = None,
        image_embeddings: List[np.ndarray] = None,
        analyzed_images: List[str] = None,
        total_cost: float = 0.0
    ):
        self.property_id = property_id
        self.detected_features = detected_features or []
        self.image_embeddings = image_embeddings or []
        self.analyzed_images = analyzed_images or []
        self.total_cost = total_cost
    
    def get_avg_embedding(self) -> Optional[np.ndarray]:
        """Average embedding across all images"""
        if not self.image_embeddings:
            return None
        return np.mean(self.image_embeddings, axis=0)
    
    def get_feature(self, name: str) -> Optional[DetectedFeature]:
        """Get feature by name"""
        for f in self.detected_features:
            if f.name.lower() == name.lower():
                return f
        return None


class VisionAnalyzer:
    """
    Hybrid vision analyzer optimized for M3 + budget
    
    Backends:
    - 'clip': Local CLIP on M3 (always used, free)
    - 'claude_vision': Claude API for feature extraction (selective use)
    """
    
    def __init__(
        self,
        claude_model: str = "claude-sonnet-4-20250514",
        mode: str = "claude_primary",  # claude_primary, qwen_primary, claude_only, qwen_only
        qwen_confidence_threshold: float = 0.7,  # Si Qwen confidence < threshold, usar Claude
        default_mode: str = "claude_description",  # "claude_description", "claude_features", "clip"
        cache_dir: Optional[str] = None,
        api_key: Optional[str] = None,
        max_claude_calls: int = 300  # Budget control
    ):
        
        self.claude_model = claude_model
        self.mode = mode
        self.qwen_confidence_threshold = qwen_confidence_threshold
        self.default_mode = default_mode
        
        # API key for Claude
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found")
        self.claude_client = Anthropic(api_key=api_key)
        
        # Cache
        self.cache_dir = Path(cache_dir or "data/cache/vision")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Budget tracking
        self.max_claude_calls = max_claude_calls
        self.claude_calls_made = 0
        self.qwen_calls_made = 0
        self.total_cost = 0.0
        
        # Lazy load models
        self._clip_model = None
        self._clip_preprocess = None
        self._qwen_analyzer = None
        
        print(f"✅ VisionAnalyzer ready (mode: {mode}, Claude budget: {max_claude_calls} calls)")
   
    @property
    def qwen_analyzer(self):
        """Lazy load Qwen2-VL"""
        if self._qwen_analyzer is None and self.mode in ['qwen_primary', 'qwen_only', 'claude_primary']:
            self._qwen_analyzer = Qwen2VisionAnalyzer(
                cache_dir=str(self.cache_dir / "qwen")
            )
        return self._qwen_analyzer

    def generate_photo_description(self,image_urls: List[str],max_images: int = 3,target_features: Optional[List[str]] = None,force_mode: Optional[str] = None) -> str:
        """
            Hybrid: genera descripción usando Claude o Qwen2 según modo
            
            Esto permite usar el mismo text_analyzer para features de texto + fotos
            
            Args:
                image_urls: URLs de imágenes
                max_images: Límite de imágenes a analizar
                target_features: Features específicos a buscar (opcional)
                force_mode: Forzar modo específico (override)
            
            Returns:
            str: Descripción en texto como si fuera parte del anuncio
        """
        mode = force_mode or self.mode
        # Select key images
        selected = self.select_key_images(
            image_urls,
            target_features or [],
            max_images
        )

        selected_urls = [url for url, _ in selected]
        
        # === MODE: qwen_only ===
        if mode == 'qwen_only':
            description, confidence = self.qwen_analyzer.generate_description(
                image_urls=selected_urls,
                max_images=max_images
            )
            self.qwen_calls_made += len(selected_urls)
            return description
        
        # === MODE: claude_only ===
        if mode == 'claude_only':
            return self._generate_description_claude(selected, target_features)
        
        # === MODE: qwen_primary ===
        if mode == 'qwen_primary':
            # Try Qwen first
            description, confidence = self.qwen_analyzer.generate_description(
                image_urls=selected_urls,
                max_images=max_images
            )
            self.qwen_calls_made += len(selected_urls)
            
            # If confidence low, fallback to Claude
            if confidence < self.qwen_confidence_threshold:
                print(f"  ⚠️  Qwen confidence {confidence:.2f} < {self.qwen_confidence_threshold}, using Claude fallback")
                return self._generate_description_claude(selected, target_features)
            
            return description
        
        # === MODE: claude_primary (default) ===
        if mode == 'claude_primary':
            # Use Claude by default
            # But if budget exhausted, use Qwen
            if self.claude_calls_made >= self.max_claude_calls:
                print(f"  ⚠️  Claude budget exhausted, using Qwen2 fallback")
                description, _ = self.qwen_analyzer.generate_description(
                    image_urls=selected_urls,
                    max_images=max_images
                )
                self.qwen_calls_made += len(selected_urls)
                return description
            
            return self._generate_description_claude(selected, target_features)
        
        # Fallback
        return self._generate_description_claude(selected, target_features)
    
    def _generate_description_claude(
        self,
        selected: List[Tuple[str, str]],
        target_features: Optional[List[str]] = None
    ) -> str:
        
        """Generate description using Claude Vision (original implementation)"""
       
        # Build prompt
        features_hint = ""
        if target_features:
            features_hint = f"\n\nPresta especial atención a: {', '.join(target_features)}"
        
        prompt = f"""Eres un agente inmobiliario profesional describiendo esta propiedad.

                    Analiza estas fotos y describe la propiedad como si la estuvieras visitando en persona.

                    IMPORTANTE - Incluye detalles sobre:
                    1. Tipo de propiedad (local comercial, piso, oficina, etc)
                    2. Acceso/entrada (¿entrada independiente desde calle? ¿portal compartido?)
                    3. Luz natural (ventanas, orientación, luminosidad)
                    4. Estado (reformado, a reformar, nuevo)
                    5. Distribución (diáfano, habitaciones, espacios)
                    6. Características visuales (terraza, balcón, vistas, etc)
                    {features_hint}

                    Responde SOLO con texto descriptivo en español, como si fuera una extensión del anuncio.
                    NO uses formato de lista, escribe en párrafos naturales."""
        
        descriptions = []
        
        for idx, (url, purpose) in enumerate(selected):
            # Check budget
            if self.claude_calls_made >= self.max_claude_calls:
                print(f"⚠️  Budget exhausted, skipping remaining images")
                break
            
            try:
                # Download image
                image = self.download_image(url)
                if image is None:
                    continue
                
                # Convert to base64
                import base64
                from io import BytesIO
                
                buffered = BytesIO()
                image.save(buffered, format="JPEG", quality=85)
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                
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
                                    "data": img_base64
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
                descriptions.append(f"[Desde foto {idx+1} - {purpose}]: {description}")
                
                # Update budget
                self.claude_calls_made += 1
                self.total_cost += 0.025
                
            except Exception as e:
                print(f"⚠️  Error processing {url}: {e}")
                continue
        
        return "\n\n".join(descriptions)

    
    @property
    def clip_model(self):
        """Lazy load CLIP model (optimized for M3)"""
        if self._clip_model is None:
            print(f"Loading CLIP model: {self.clip_model_name} on M3...")
            import torch
            import clip
            
            # M3 optimization: use MPS (Metal Performance Shaders)
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            print(f"Using device: {device}")
            
            self._clip_model, self._clip_preprocess = clip.load(
                self.clip_model_name, 
                device=device
            )
            print("✅ CLIP loaded on M3")
        
        return self._clip_model, self._clip_preprocess
    
    def select_key_images(
        self,
        image_urls: List[str],
        query_requirements: List[str],
        max_images: int = 5
    ) -> List[Tuple[str, str]]:
        """
        Smart image selection based on query needs
        
        Returns: List[(url, purpose)]
        
        Heuristics:
        - First 2 images: Usually exterior/facade (entrada_independiente)
        - Middle images: Interior views (luz_natural, layout)
        - Images with keywords in URL/filename
        - Max N images to control cost
        """
        selected = []
        
        # Priority 1: First 2 (exterior likely)
        if len(image_urls) >= 2:
            selected.append((image_urls[0], "exterior_main"))
            selected.append((image_urls[1], "exterior_secondary"))
        elif len(image_urls) >= 1:
            selected.append((image_urls[0], "main_view"))
        
        # Priority 2: Keyword-based selection
        keywords = {
            "entrada": "entrance",
            "fachada": "facade",
            "salon": "living_room",
            "interior": "interior",
            "terraza": "terrace",
            "cocina": "kitchen"
        }
        
        for url in image_urls[2:]:
            if len(selected) >= max_images:
                break
            
            url_lower = url.lower()
            for keyword, purpose in keywords.items():
                if keyword in url_lower:
                    selected.append((url, purpose))
                    break
        
        # Fill remaining slots with evenly distributed images
        remaining = max_images - len(selected)
        if remaining > 0 and len(image_urls) > len(selected):
            available = [u for u in image_urls if not any(u == s[0] for s in selected)]
            step = max(1, len(available) // remaining)
            for i in range(0, len(available), step):
                if len(selected) >= max_images:
                    break
                selected.append((available[i], f"view_{len(selected)}"))
        
        return selected[:max_images]
    
    def download_image(self, url: str) -> Optional[Image.Image]:
        """Download and open image"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            print(f"⚠️  Failed to download {url}: {e}")
            return None
    
    def generate_clip_embedding(
        self,
        image_url: str,
        property_id: str,
        image_idx: int = 0
    ) -> Optional[np.ndarray]:
        """
        Generate CLIP embedding for image (cached)
        FREE - runs on M3
        """
        cache_file = self.cache_dir / f"{property_id}_img{image_idx}_clip.npy"
        if cache_file.exists():
            return np.load(cache_file)
        
        try:
            # Load model
            model, preprocess = self.clip_model
            import torch
            
            # Download and preprocess image
            image = self.download_image(image_url)
            if image is None:
                return None
            
            image_input = preprocess(image).unsqueeze(0)
            
            # Move to M3
            device = next(model.parameters()).device
            image_input = image_input.to(device)
            
            # Generate embedding
            with torch.no_grad():
                embedding = model.encode_image(image_input)
                embedding = embedding.cpu().numpy().flatten()
            
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            
            # Cache
            np.save(cache_file, embedding)
            return embedding
            
        except Exception as e:
            print(f"⚠️  CLIP embedding error for {image_url}: {e}")
            return None
    
    def compute_image_text_similarity(
        self,
        image_embedding: np.ndarray,
        text: str
    ) -> float:
        """
        Compute similarity between image embedding and text
        Uses CLIP's text encoder
        """
        try:
            model, _ = self.clip_model
            import torch
            import clip
            
            # Encode text
            device = next(model.parameters()).device
            text_tokens = clip.tokenize([text]).to(device)
            
            with torch.no_grad():
                text_embedding = model.encode_text(text_tokens)
                text_embedding = text_embedding.cpu().numpy().flatten()
            
            # Normalize
            text_embedding = text_embedding / np.linalg.norm(text_embedding)
            
            # Cosine similarity
            similarity = float(np.dot(image_embedding, text_embedding))
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            print(f"⚠️  Similarity computation error: {e}")
            return 0.0
    
    def extract_features_with_claude(
        self,
        image_url: str,
        query_text: str,
        target_features: Optional[List[str]] = None
    ) -> List[DetectedFeature]:
        """
        Dynamic feature extraction using Claude Vision
        COSTS MONEY - use strategically!
        
        Returns: List[DetectedFeature] with same schema as text analyzer
        """
        # Budget check
        if self.claude_calls_made >= self.max_claude_calls:
            print(f"⚠️  Budget exhausted ({self.max_claude_calls} calls)")
            return []
        
        try:
            # Download image and convert to base64
            image = self.download_image(image_url)
            if image is None:
                return []
            
            import base64
            from io import BytesIO
            
            buffered = BytesIO()
            image.save(buffered, format="JPEG", quality=85)
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Construct prompt for dynamic feature extraction
            target_features_str = ", ".join(target_features) if target_features else "cualquier característica relevante"
            
            prompt = f"""Analiza esta imagen de una propiedad inmobiliaria.

Query del usuario: "{query_text}"

Detecta características visuales relevantes para este query. Busca especialmente: {target_features_str}

Ejemplos de características a detectar:
- entrada_independiente: puerta directa desde la calle
- luz_natural: ventanas, ventanales, claridad
- terraza_privada: espacio exterior privado
- espacio_diafano: espacio abierto sin divisiones
- estado_reforma: necesita renovación o está reformado
- cualquier otra característica visual relevante

Responde SOLO con JSON válido (sin markdown):
{{
  "detected_features": [
    {{
      "name": "nombre_en_snake_case",
      "value": "descripción breve o medida si aplica",
      "confidence": 0.0-1.0,
      "evidence": "qué ves en la imagen que justifica esto"
    }}
  ]
}}

Si no detectas ninguna característica relevante, devuelve lista vacía."""

            # Call Claude Vision API
            response = self.claude_client.messages.create(
                model=self.claude_model,
                max_tokens=1500,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": img_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }]
            )
            
            # Parse response
            response_text = response.content[0].text.strip()
            
            # Clean markdown if present
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            response_text = response_text.strip()
            
            result = json.loads(response_text)
            features = [
                DetectedFeature(
                    **f,
                    source="image_claude"
                )
                for f in result.get("detected_features", [])
            ]
            
            # Update budget tracking
            self.claude_calls_made += 1
            self.total_cost += 0.025  # Approximate cost per image
            
            return features
            
        except Exception as e:
            print(f"⚠️  Claude Vision error: {e}")
            return []
    
    def analyze_property_stage1(
        self,
        property_id: str,
        image_urls: List[str],
        max_images: int = 5
    ) -> VisionAnalysis:
        """
        Stage 1: Fast CLIP embeddings for ALL properties
        FREE - runs on M3
        
        Returns: VisionAnalysis with embeddings only
        """
        print(f"  Stage 1 (CLIP): {property_id}")
        
        # Select key images
        selected = self.select_key_images(image_urls, [], max_images)
        
        # Generate embeddings
        embeddings = []
        analyzed = []
        
        for idx, (url, purpose) in enumerate(selected):
            embedding = self.generate_clip_embedding(url, property_id, idx)
            if embedding is not None:
                embeddings.append(embedding)
                analyzed.append(url)
        
        return VisionAnalysis(
            property_id=property_id,
            image_embeddings=embeddings,
            analyzed_images=analyzed,
            total_cost=0.0
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
        Stage 2: Detailed Claude Vision analysis for TOP candidates
        COSTS MONEY - use only on filtered properties!
        
        Args:
            max_images: Limit to control cost (2 images = €0.05)
        """
        print(f"  Stage 2 (Claude): {property_id}")
        
        # Select key images (fewer for cost)
        selected = self.select_key_images(image_urls, target_features, max_images)
        
        all_features = []
        analyzed = []
        
        for url, purpose in selected:
            features = self.extract_features_with_claude(url, query_text, target_features)
            all_features.extend(features)
            analyzed.append(url)
        
        # Deduplicate features (keep highest confidence)
        unique_features = {}
        for f in all_features:
            if f.name not in unique_features or f.confidence > unique_features[f.name].confidence:
                unique_features[f.name] = f
        
        return VisionAnalysis(
            property_id=property_id,
            detected_features=list(unique_features.values()),
            analyzed_images=analyzed,
            total_cost=len(selected) * 0.025
        )
    
    def get_budget_status(self) -> dict:
        """Check remaining budget"""
        return {
            'mode': self.mode,
            "claude_calls_made": self.claude_calls_made,
            "claude_max_calls": self.max_claude_calls,
            "claude_remaining": self.max_claude_calls - self.claude_calls_made,
            'qwen_calls_made': self.qwen_calls_made,
            "total_cost_eur": self.total_cost,
            "remaining_budget_eur": (self.max_claude_calls - self.claude_calls_made) * 0.025
        }