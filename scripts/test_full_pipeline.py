#!/usr/bin/env python3
"""
Test completo del pipeline: Query Parser → Text Analyzer → Matching

Uso:
    python scripts/test_full_pipeline.py "Local con entrada independiente máx 250k"
"""
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.query_parser.parser import QueryParser
from src.property_analysis.text_analyzer import PropertyTextAnalyzer
from src.property_analysis.schemas import QueryRequirement

console = Console()

def main():
    if len(sys.argv) < 2:
        console.print("❌ Uso: python scripts/test_full_pipeline.py 'tu query aquí'", style="bold red")
        console.print("\nEjemplos:")
        console.print("  python scripts/test_full_pipeline.py 'Local Barcelona entrada independiente máx 250k'")
        console.print("  python scripts/test_full_pipeline.py 'Piso luminoso 3 habitaciones'")
        sys.exit(1)
    
    query = " ".join(sys.argv[1:])
    
    console.print(Panel.fit(f"🔍 Query: {query}", style="bold blue"))
    
    # 1. Parse Query
    console.print("\n[bold cyan]STEP 1: Parsing Query[/bold cyan]")
    query_parser = QueryParser()
    parsed = query_parser.parse(query)
    
    console.print(f"  Direct filters: {parsed.direct_filters.model_dump(exclude_none=True)}")
    console.print(f"  Indirect filters: {parsed.indirect_filters.model_dump(exclude_none=True)}")
    console.print(f"  Confidence: {parsed.confidence:.2f}")
    
    # Convertir indirect filters a requirements
    requirements = []
    if parsed.indirect_filters.entrance_type:
        requirements.append(QueryRequirement(
            feature_name="entrada_independiente",
            importance=1.0,
            required=True
        ))
    if parsed.indirect_filters.natural_light:
        requirements.append(QueryRequirement(
            feature_name="luz_natural",
            importance=0.8
        ))
    if parsed.indirect_filters.features:
        for feature in parsed.indirect_filters.features:
            requirements.append(QueryRequirement(
                feature_name=feature,
                importance=0.6
            ))
    
    # 2. Cargar propiedades
    console.print("\n[bold cyan]STEP 2: Loading Properties[/bold cyan]")
    data_dir = Path("data/raw")
    json_files = sorted(data_dir.glob("fotocasa_*.json"))
    
    all_properties = []
    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)
            all_properties.extend(data.get('properties', []))
    
    console.print(f"  Loaded {len(all_properties)} properties from {len(json_files)} files")
    
    # Filtrar por direct filters (simple)
    filtered_properties = all_properties
    if parsed.direct_filters.max_price:
        filtered_properties = [p for p in filtered_properties 
                              if p.get('price', 0) <= parsed.direct_filters.max_price]
    if parsed.direct_filters.property_type:
        # Simple check en título o descripción
        prop_type = parsed.direct_filters.property_type
        filtered_properties = [p for p in filtered_properties 
                              if prop_type in p.get('title', '').lower() or 
                                 prop_type in p.get('description', '').lower()]
    
    console.print(f"  After direct filters: {len(filtered_properties)} properties")
    
    if not filtered_properties:
        console.print("\n❌ No properties match direct filters", style="bold red")
        sys.exit(0)
    
    # Limitar a top 20 para análisis
    properties_to_analyze = filtered_properties[:5]
    console.print(f"  Analyzing top {len(properties_to_analyze)} properties")
    
    # 3. Analizar con Text Analyzer
    console.print("\n[bold cyan]STEP 3: Analyzing Properties (backend: api)[/bold cyan]")
    analyzer = PropertyTextAnalyzer(backend="api")
    
    analyses = []
    with console.status("[bold green]Analyzing properties...") as status:
        for i, prop in enumerate(properties_to_analyze, 1):
            status.update(f"[bold green]Analyzing {i}/{len(properties_to_analyze)}...")
            analysis = analyzer.analyze(
                property_id=prop['id'],
                description=prop.get('description', ''),
                generate_embedding=True
            )
            analyses.append(analysis)
    
    console.print(f"  ✅ Analyzed {len(analyses)} properties")
    
    # 4. Match contra query
    console.print("\n[bold cyan]STEP 4: Matching Properties[/bold cyan]")
    matches = []
    for analysis in analyses:
        match = analyzer.match_against_query(
            analysis,
            query,
            requirements if requirements else None
        )
        matches.append((match, analysis))
    
    # Ordenar por score
    matches.sort(key=lambda x: x[0].final_score, reverse=True)
    
    # 5. Mostrar resultados
    console.print("\n" + "="*80)
    console.print("[bold green]TOP 10 MATCHES[/bold green]")
    console.print("="*80 + "\n")
    
    for i, (match, analysis) in enumerate(matches[:10], 1):
        # Encontrar property original
        prop = next(p for p in properties_to_analyze if p['id'] == match.property_id)
        
        # Determinar calidad del match
        if match.final_score >= 0.7:
            match_quality = "[bold green]EXCELLENT MATCH ⭐⭐⭐[/bold green]"
        elif match.final_score >= 0.5:
            match_quality = "[yellow]GOOD MATCH ⭐⭐[/yellow]"
        elif match.final_score >= 0.3:
            match_quality = "[orange3]FAIR MATCH ⭐[/orange3]"
        else:
            match_quality = "[red]WEAK MATCH[/red]"
        
        console.print(f"[bold]#{i}. Property {prop['id']}[/bold] - {match_quality}")
        console.print(f"   📍 {prop['location']}")
        console.print(f"   💰 {prop['price']:,}€ | {prop.get('size_m2', 'N/A')}m² | {prop.get('rooms', 'N/A')} rooms")
        
        # Scores
        console.print(f"\n   📊 Scores:")
        console.print(f"      Final Score:      {match.final_score:.2f}")
        console.print(f"      Feature Match:    {match.feature_match_score:.2f}")
        console.print(f"      Semantic Similar: {match.semantic_similarity_score:.2f}")
        console.print(f"      Quality Score:    {analysis.overall_quality_score:.2f}")
        
        # Features matched
        if match.matched_features:
            console.print(f"\n   ✅ Matched Features:")
            for feat in match.matched_features[:5]:
                bar = "█" * int(feat.confidence * 10)
                console.print(f"      {feat.name:.<30} {feat.confidence:.2f} {bar}")
        
        # Missing
        if match.missing_requirements:
            console.print(f"\n   ⚠️  Missing: {', '.join(match.missing_requirements)}")
        
        # Top features detectados
        top_features = sorted(analysis.detected_features, 
                            key=lambda f: f.confidence, reverse=True)[:3]
        if top_features:
            console.print(f"\n   🏷️  Top Features:")
            for feat in top_features:
                console.print(f"      • {feat.name} ({feat.confidence:.2f})")
        
        # Description preview
        desc = prop.get('description', 'N/A')[:150]
        console.print(f"\n   📝 Description: {desc}...")
        console.print(f"   🔗 {prop['url']}\n")
        console.print("   " + "-"*70 + "\n")
    
    # 6. Estadísticas generales
    console.print("\n" + "="*80)
    console.print("[bold cyan]STATISTICS[/bold cyan]")
    console.print("="*80 + "\n")
    
    scores = [m[0].final_score for m in matches]
    feature_scores = [m[0].feature_match_score for m in matches]
    semantic_scores = [m[0].semantic_similarity_score for m in matches]
    
    console.print(f"  Average Final Score:    {sum(scores)/len(scores):.2f}")
    console.print(f"  Average Feature Score:  {sum(feature_scores)/len(feature_scores):.2f}")
    console.print(f"  Average Semantic Score: {sum(semantic_scores)/len(semantic_scores):.2f}")
    console.print(f"\n  Excellent matches (≥0.7): {sum(1 for s in scores if s >= 0.7)}")
    console.print(f"  Good matches (≥0.5):      {sum(1 for s in scores if s >= 0.5)}")
    console.print(f"  Fair matches (≥0.3):      {sum(1 for s in scores if s >= 0.3)}")
    console.print(f"  Weak matches (<0.3):      {sum(1 for s in scores if s < 0.3)}")

if __name__ == "__main__":
    main()