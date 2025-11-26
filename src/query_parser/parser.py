"""
Query Parser usando Claude API
"""
import json
import os
from anthropic import Anthropic
from typing import Optional
import mlflow

from dotenv import load_dotenv
load_dotenv()

from .schemas import ParsedQuery
from .prompts import get_prompt

class QueryParser:
    def __init__(
        self, 
        model: str = "claude-sonnet-4-20250514",
        prompt_version: str = "v1.0",
        api_key: Optional[str] = None
    ):
        self.model = model
        self.prompt_version = prompt_version
        self.prompt_config = get_prompt(prompt_version)
        
        # Cliente Anthropic
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        self.client = Anthropic(api_key=api_key)
    
    def parse(self, query: str, log_to_mlflow: bool = False) -> ParsedQuery:
        """
        Parsea una query en lenguaje natural
        
        Args:
            query: Query del usuario en español
            log_to_mlflow: Si True, logea a MLflow
            
        Returns:
            ParsedQuery con filtros estructurados
        """
        # Preparar prompts
        system_prompt = self.prompt_config["system"]
        user_prompt = self.prompt_config["user_template"].format(query=query)
        
        # Llamada a Claude
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        
        # Extraer JSON
        response_text = response.content[0].text.strip()
        
        # Limpiar markdown si existe
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        response_text = response_text.strip()
        
        # Parse JSON
        try:
            parsed_dict = json.loads(response_text)
            result = ParsedQuery(**parsed_dict)
        except Exception as e:
            # Fallback: query inválida
            result = ParsedQuery(
                original_query=query,
                direct_filters={},
                indirect_filters={},
                confidence=0.0
            )
        
        # Log a MLflow si se solicita
        if log_to_mlflow and mlflow.active_run():
            mlflow.log_param("query", query)
            mlflow.log_param("prompt_version", self.prompt_version)
            mlflow.log_metric("confidence", result.confidence)
            mlflow.log_dict(result.dict(), "parsed_query.json")
        
        return result
    
    def parse_batch(self, queries: list[str]) -> list[ParsedQuery]:
        """Parsea múltiples queries"""
        return [self.parse(q) for q in queries]