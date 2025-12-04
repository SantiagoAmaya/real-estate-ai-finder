"""
Vision Decision Agent: Decide inteligentemente cuándo usar análisis visual

Usa LLM para evaluar si un query requiere análisis de imágenes.
"""
from anthropic import Anthropic
import os
from dotenv import load_dotenv
from typing import Dict, List
import json

load_dotenv()


class VisionDecisionAgent:
    """Agente que decide si un query necesita análisis visual"""
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str = None
    ):
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = Anthropic(api_key=api_key)
        self.model = model
        
        self.decision_prompt = """
            Eres un agente experto en intención de búsqueda inmobiliaria. Tu objetivo es determinar la DISCREPANCIA entre lo que busca el usuario y lo que los metadatos de texto pueden ofrecer.

            Tu tarea: Decidir si el query requiere una validación visual (fotos/imágenes) o si los filtros de datos estructurados (texto) son suficientes.

            PRINCIPIOS DE DECISIÓN:

            REQUIERE VISION (return True) si el query implica:
            1. Juicios Estéticos o Subjetivos: Solicitudes que dependen del gusto personal, estilo, atmósfera o "vibe" (ej. sensaciones, estilos arquitectónicos, calidad de la luz).
            2. Verificación de Estado/Condición: Cuando se necesita confirmar el deterioro, la calidad de los acabados o el nivel de reforma real más allá de lo declarado.
            3. Comprensión Espacial/Distribución: Dudas sobre cómo fluye el espacio, la amplitud real (más allá de los m²), o la relación entre estancias que no se explica con números.
            4. Elementos del Entorno: Vistas específicas, orientación solar real, o características del exterior inmediato.

            NO REQUIERE VISION (return False) si el query se basa en:
            1. Datos Cuantitativos/Estructurados: Todo lo que sea contable o medible (precio, superficie, nro. habitaciones, planta).
            2. Categorías Binarias: Existencia o no de elementos estándar ("con ascensor", "con garaje", "cerca del metro").
            3. Ubicación Geográfica: Barrios, calles o zonas específicas definibles en un mapa.

            Query del usuario: "{query}"

            Responde SOLO con JSON válido (sin markdown):
            {{
            "needs_vision": true/false,
            "confidence": 0.0-1.0,
            "reasoning": "Explica si la solicitud es subjetiva/visual o objetiva/dato",
            "key_visual_features": ["lista abstracta de qué buscar en la imagen", "ej: calidad de la luz", "ej: estilo de cocina"]
            }}
            """
    
    def should_use_vision(
        self,
        query: str,
        text_score: float = None,
        top_candidates_scores: List[float] = None
    ) -> Dict:
        """
        Decide si usar análisis visual
        
        Returns:
            {
                'use_vision': bool,
                'confidence': float,
                'reasoning': str,
                'trigger_reason': str,
                'visual_features': List[str]
            }
        """
        # Validate inputs
        if top_candidates_scores is None:
            top_candidates_scores = []

        # Regla 1: Si score muy bajo o muy alto, no necesita vision
        if text_score is not None:
            if text_score < 0.2:
                return {
                    'use_vision': False,
                    'confidence': 0.95,
                    'reasoning': 'Text score too low - property clearly irrelevant',
                    'trigger_reason': 'low_score_shortcut',
                    'visual_features': []
                }
            
            if text_score > 0.8:
                return {
                    'use_vision': False,
                    'confidence': 0.9,
                    'reasoning': 'Text score very high - already excellent match',
                    'trigger_reason': 'high_score_shortcut',
                    'visual_features': []
                }
        
        # Regla 2: Empates - usar vision para desempatar
        if top_candidates_scores and len(top_candidates_scores) >= 2:
            try:
                top_diff = abs(float(top_candidates_scores[0]) - float(top_candidates_scores[1]))
                if top_diff < 0.05:  # Empate técnico
                    return {
                        'use_vision': True,
                        'confidence': 0.85,
                        'reasoning': 'Tie-breaking between similar candidates',
                        'trigger_reason': 'tie_breaking',
                        'visual_features': []
                    }
            except Exception as e:
                print(f"⚠️  Error evaluating tie-breaking: {e}")
                pass
        
        # Regla 3: Preguntar al LLM
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                messages=[{
                    "role": "user",
                    "content": self.decision_prompt.format(query=query)
                }]
            )
            
            response_text = response.content[0].text.strip()
            
            # Limpiar markdown si existe
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            response_text = response_text.strip()
            
            result = json.loads(response_text)
            
            return {
                'use_vision': result['needs_vision'],
                'confidence': result['confidence'],
                'reasoning': result['reasoning'],
                'trigger_reason': 'llm_decision',
                'visual_features': result.get('key_visual_features', [])
            }
            
        except Exception as e:
            # Fallback: decisión conservadora
            print(f"⚠️  Vision decision error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'use_vision': True,  # Conservative - when in doubt, use vision
                'confidence': 0.5,
                'reasoning': f'Error in LLM decision: {str(e)}',
                'trigger_reason': 'error_fallback',
                'visual_features': []
            }
    
    def batch_decide(
        self,
        query: str,
        properties_with_scores: List[Dict]
    ) -> List[bool]:
        """
        Decide para múltiples propiedades eficientemente
        
        Args:
            properties_with_scores: [{'property_id': ..., 'text_score': ...}, ...]
        
        Returns:
            List[bool] - True = usar vision, False = skip
        """
        decisions = []
        
        # Get top scores for tie-breaking
        scores = [p['text_score'] for p in properties_with_scores]
        scores_sorted = sorted(scores, reverse=True)
        
        for prop_data in properties_with_scores:
            decision = self.should_use_vision(
                query=query,
                text_score=prop_data['text_score'],
                top_candidates_scores=scores_sorted[:5]
            )
            decisions.append(decision['use_vision'])
        
        return decisions