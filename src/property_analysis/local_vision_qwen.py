"""
Local Vision Analysis usando Qwen2-VL en RTX 5090

Alternativa gratuita y local a Claude Vision.
Calidad: ⭐⭐⭐⭐ (muy buena para ser local)
Velocidad: ~4s por imagen en RTX 5090
"""
from typing import List, Optional, Tuple
from pathlib import Path
import torch
from PIL import Image
import requests
from io import BytesIO
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class Qwen2VisionAnalyzer:
    """Análisis de visión local con Qwen2-VL"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        cache_dir: Optional[str] = None,
        device: str = "auto"
    ):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir or "data/cache/qwen")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Lazy load
        self._model = None
        self._processor = None
        self.device = device
        
        print(f"✅ Qwen2VisionAnalyzer initialized (model will load on first use)")
    
    @property
    def model(self):
        """Lazy load model (only when needed)"""
        if self._model is None:
            print(f"Loading Qwen2-VL model: {self.model_name} on RTX 5090...")
            
            # Check device (CUDA for Linux, MPS for Mac, CPU fallback)
            if torch.cuda.is_available():
                device_type = "cuda"
                device_name = torch.cuda.get_device_name(0)
            elif torch.backends.mps.is_available():
                device_type = "mps"
                device_name = "Apple Silicon (MPS)"
            else:
                device_type = "cpu"
                device_name = "CPU (slow)"
           
            print(f"✅ Using: {device_type} - {device_name}")
            
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
                cache_dir=str(self.cache_dir)
            )
            
            self._processor = AutoProcessor.from_pretrained(
                self.model_name,
                cache_dir=str(self.cache_dir)
            )
            
            print("✅ Qwen2-VL loaded successfully")
        
        return self._model, self._processor
    
    def download_image(self, url: str) -> Optional[Image.Image]:
        """Download image from URL"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            print(f"⚠️  Failed to download {url}: {e}")
            return None
    
    def generate_description(
        self,
        image_urls: List[str],
        prompt_template: Optional[str] = None,
        max_images: int = 3,
        max_new_tokens: int = 400
    ) -> Tuple[str, float]:
        """
        Generate property description from images
        
        Returns:
            (description: str, confidence: float)
        """
        model, processor = self.model
        
        if prompt_template is None:
            prompt_template = """Eres un agente inmobiliario profesional. Describe esta propiedad en español.

Incluye:
1. Tipo de propiedad (local comercial, piso, oficina, etc)
2. Acceso/entrada (¿entrada independiente desde calle? ¿portal compartido?)
3. Luz natural (ventanas, orientación, luminosidad)
4. Estado (reformado, a reformar, nuevo)
5. Distribución (diáfano, habitaciones, espacios)
6. Características visuales importantes

Escribe en párrafos naturales, como extensión del anuncio."""
        
        descriptions = []
        confidences = []
        
        for idx, url in enumerate(image_urls[:max_images]):
            try:
                # Download image
                image = self.download_image(url)
                if image is None:
                    continue
                
                # Prepare messages (Qwen2-VL format)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image
                            },
                            {
                                "type": "text",
                                "text": prompt_template
                            }
                        ]
                    }
                ]
                
                # Prepare for generation
                text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                image_inputs, video_inputs = process_vision_info(messages)
                
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                )
                
                # Auto-detect device
                if torch.cuda.is_available():
                    device = "cuda"
                elif torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
                
                inputs = inputs.to(device)
                
                # Generate
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False  # Deterministic
                    )
                
                # Decode
                generated_ids = [
                    output_ids[len(input_ids):]
                    for input_ids, output_ids in zip(inputs.input_ids, outputs)
                ]
                
                description = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]
                
                descriptions.append(f"[Imagen {idx+1}]: {description.strip()}")
                
                # Estimate confidence (heuristic based on length and coherence)
                confidence = min(1.0, len(description.split()) / 50.0)  # More words = more confident
                if len(description.split()) < 20:
                    confidence *= 0.5  # Penalize very short descriptions
                
                confidences.append(confidence)
                
            except Exception as e:
                print(f"⚠️  Error processing image {url}: {e}")
                continue
        
        if not descriptions:
            return "", 0.0
        
        combined_description = "\n\n".join(descriptions)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return combined_description, avg_confidence
    
    def get_status(self) -> dict:
        """Get analyzer status"""
        return {
            'model_loaded': self._model is not None,
            'model_name': self.model_name,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }