#!/usr/bin/env python3
"""
Analiza una propiedad específica

Uso:
    python scripts/analyze_property.py <property_id>
"""
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.property_analysis.text_analyzer import PropertyTextAnalyzer

def main():
    if len(sys.argv) < 2:
        print("❌ Uso: python scripts/analyze_property.py <property_id>")
        print("   O: python scripts/analyze_property.py all  # Analiza todas")
        sys.exit(1)
    
    property_id = sys.argv[1]
    
    # Cargar datos
    data_dir = Path("data/raw")
    latest_file = sorted(data_dir.glob("fotocasa_*.json"))[-1]
    
    with open(latest_file) as f:
        data = json.load(f)
    
    properties = data['properties']
    
    # Filtrar o analizar todas
    if property_id == "all":
        props_to_analyze = properties
    else:
        props_to_analyze = [p for p in properties if p['id'] == property_id]
        if not props_to_analyze:
            print(f"❌ Property {property_id} not found")
            sys.exit(1)
    
    # Analizar
    analyzer = PropertyTextAnalyzer()
    
    print(f"🔍 Analyzing {len(props_to_analyze)} properties...\n")
    
    for prop in props_to_analyze:
        result = analyzer.analyze(
            property_id=prop['id'],
            description=prop.get('description', '')
        )
        
        print(f"{'='*70}")
        print(f"🏠 Property: {prop['id']}")
        print(f"📍 {prop['location']}")
        print(f"💰 {prop['price']:,}€ | {prop.get('size_m2', 'N/A')}m²")
        print(f"🔗 {prop['url']}")
        
        detected = result.features.get_detected_features(threshold=0.5)
        
        if detected:
            print(f"\n✅ Detected Features:")
            for feature, score in sorted(detected.items(), key=lambda x: x[1], reverse=True):
                bars = "█" * int(score * 10)
                print(f"   {feature:.<30} {score:.2f} {bars}")
        else:
            print(f"\n⚠️  No clear features detected")
        
        print(f"\n📊 Confidence: {result.features.confidence:.2f}")
        print()

if __name__ == "__main__":
    main()