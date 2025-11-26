#!/usr/bin/env python3
"""
Búsqueda de propiedades usando query parser + scraper

Uso:
    python scripts/search_properties.py "Local Barcelona entrada independiente máx 250k"
"""
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.query_parser.parser import QueryParser
from src.data.scraper import FotocasaScraper

def main():
    if len(sys.argv) < 2:
        print("❌ Uso: python scripts/search_properties.py 'tu query aquí'")
        sys.exit(1)
    
    query = " ".join(sys.argv[1:])
    
    print(f"🔍 Query: {query}\n")
    
    # 1. Parse query
    parser = QueryParser()
    parsed = parser.parse(query)
    
    print("📊 Parsed Query:")
    print(f"   Direct filters: {parsed.direct_filters.model_dump(exclude_none=True)}")
    print(f"   Indirect filters: {parsed.indirect_filters.model_dump(exclude_none=True)}")
    print(f"   Confidence: {parsed.confidence:.2f}\n")
    
    # 2. Scrape con direct filters
    scraper = FotocasaScraper()
    
    # Convertir a parámetros del scraper
    scraper_params = {
        "location": parsed.direct_filters.location or "barcelona-capital",
        "property_type": parsed.direct_filters.property_type or "vivienda",
        "operation": parsed.direct_filters.operation or "comprar",
        "max_pages": 2
    }
    
    if parsed.direct_filters.min_price:
        scraper_params["min_price"] = parsed.direct_filters.min_price
    if parsed.direct_filters.max_price:
        scraper_params["max_price"] = parsed.direct_filters.max_price
    if parsed.direct_filters.min_size_m2:
        scraper_params["min_size_m2"] = parsed.direct_filters.min_size_m2
    if parsed.direct_filters.min_rooms:
        scraper_params["min_rooms"] = parsed.direct_filters.min_rooms
    
    print(f"🚀 Scraping with: {scraper_params}\n")
    
    result = scraper.scrape_properties(**scraper_params)
    
    print(f"✅ Found {len(result.properties)} properties")
    
    if result.properties:
        print("\n📋 Sample results:")
        for i, prop in enumerate(result.properties[:3], 1):
            print(f"\n{i}. {prop.price:,}€ | {prop.size_m2}m² | {prop.location}")
            print(f"   {prop.url}")
    
    # TODO Fase 3: Aplicar indirect_filters con NLP/CV
    print("\n⏭️  Indirect filters will be applied in Phase 3 (NLP/CV analysis)")

if __name__ == "__main__":
    main()