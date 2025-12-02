#!/usr/bin/env python3
"""
Analiza propiedades y matchea contra query

Uso:
    python scripts/analyze_and_match.py "Local con cocina equipada"
"""
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.property_analysis.text_analyzer import PropertyTextAnalyzer
from src.property_analysis.schemas import QueryRequirement

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="+", help="Search query")
    parser.add_argument("--backend", choices=["api", "local"], default="api",
                       help="Embedding backend (default: api)")
    args = parser.parse_args()
    
    query = " ".join(args.query)
    if len(sys.argv) < 2:
        print("❌ Uso: python scripts/analyze_and_match.py 'tu query aquí'")
        sys.exit(1)
    
    query = " ".join(sys.argv[1:])
    
    # Cargar datos
    data_dir = Path("data/raw")
    latest_file = sorted(data_dir.glob("fotocasa_*.json"))[-1]
    
    with open(latest_file) as f:
        data = json.load(f)
    
    properties = data['properties'][:10]  # Primeras 10
    
    print(f"🔍 Query: {query}")
    print(f"📊 Analyzing {len(properties)} properties...\n")
    
    # Analizar
    analyzer = PropertyTextAnalyzer(backend=args.backend)
    print(f"🔧 Using backend: {args.backend}")
    analyses = analyzer.analyze_batch(properties, generate_embeddings=True)
    
    # Match
    matches = []
    for analysis in analyses:
        match = analyzer.match_against_query(analysis, query)
        matches.append((match, analysis))
    
    # Ordenar por score
    matches.sort(key=lambda x: x[0].final_score, reverse=True)
    
    # Mostrar top 5
    print(f"\n{'='*70}")
    print("TOP 5 MATCHES\n")
    
    for i, (match, analysis) in enumerate(matches[:5], 1):
        prop = next(p for p in properties if p['id'] == match.property_id)
        
        print(f"{i}. Property: {prop['id']}")
        print(f"   📍 {prop['location']}")
        print(f"   💰 {prop['price']:,}€ | {prop.get('size_m2', 'N/A')}m²")
        print(f"   🔗 {prop['url']}")
        print(f"\n   📊 Scores:")
        print(f"      Final: {match.final_score:.2f}")
        print(f"      Features: {match.feature_match_score:.2f}")
        print(f"      Semantic: {match.semantic_similarity_score:.2f}")
        
        if match.matched_features:
            print(f"\n   ✅ Matched Features:")
            for feat in match.matched_features[:3]:
                print(f"      - {feat.name} ({feat.confidence:.2f})")
        
        if match.missing_requirements:
            print(f"\n   ⚠️  Missing: {', '.join(match.missing_requirements)}")
        
        print()

if __name__ == "__main__":
    main()