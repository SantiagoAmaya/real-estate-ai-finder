#!/usr/bin/env python3
"""
Test end-to-end completo: Query → Scrape → Analyze → Match

Uso:
    python scripts/end_to_end_test.py "Local Barcelona entrada independiente máx 250k"
    python scripts/end_to_end_test.py "Piso luminoso 3 habitaciones" --skip-scrape
"""
import sys
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
import argparse

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.query_parser.parser import QueryParser
from src.data.scraper import FotocasaScraper
from src.property_analysis.text_analyzer import PropertyTextAnalyzer
from src.property_analysis.schemas import QueryRequirement
from src.data_quality.validator import FotocasaValidator

console = Console()

def parse_arguments():
    parser = argparse.ArgumentParser(description="End-to-end property search test")
    parser.add_argument("query", nargs="+", help="Search query in Spanish")
    parser.add_argument("--skip-scrape", action="store_true", 
                       help="Skip scraping, use existing data")
    parser.add_argument("--backend", choices=["api", "local"], default="api",
                       help="Text analyzer backend (default: api)")
    parser.add_argument("--max-results", type=int, default=10,
                       help="Max properties to show (default: 10)")
    return parser.parse_args()

def main():
    args = parse_arguments()
    query = " ".join(args.query)
    
    console.print(Panel.fit(f"🚀 END-TO-END TEST\nQuery: {query}", style="bold blue"))
    
    # ============================================================
    # STEP 1: Parse Query
    # ============================================================
    console.print("\n[bold cyan]STEP 1: Parsing Query[/bold cyan]")
    query_parser = QueryParser()
    parsed = query_parser.parse(query)
    
    console.print(f"  ✅ Direct filters: {parsed.direct_filters.model_dump(exclude_none=True)}")
    console.print(f"  ✅ Indirect filters: {parsed.indirect_filters.model_dump(exclude_none=True)}")
    console.print(f"  ✅ Confidence: {parsed.confidence:.2f}")
    
    # Convert to scraper params
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
    
    # ============================================================
    # STEP 2: Scrape (or load existing)
    # ============================================================
    console.print("\n[bold cyan]STEP 2: Getting Properties[/bold cyan]")
    
    if not args.skip_scrape:
        console.print(f"  🔍 Scraping with params: {scraper_params}")
        
        scraper = FotocasaScraper()
        result = scraper.scrape_properties(**scraper_params)
        
        properties = result.properties
        
        if properties:
            # Save results
            filepath = result.save("data/raw")
            console.print(f"  ✅ Scraped {len(properties)} properties")
            console.print(f"  💾 Saved to: {filepath.name}")
            
            # Validate
            validator = FotocasaValidator()
            validation = validator.validate_json(filepath)
            
            if validation['passed']:
                console.print(f"  ✅ Data quality: PASSED")
            else:
                console.print(f"  ⚠️  Data quality: {validation['passed_count']}/{validation['total_checks']} checks passed")
        else:
            console.print(f"  ❌ No properties found!")
            return
    else:
        console.print(f"  📂 Loading existing data...")
        
        # Load most recent matching data
        data_dir = Path("data/raw")
        json_files = sorted(data_dir.glob("fotocasa_*.json"), reverse=True)
        
        properties = []
        for json_file in json_files[:3]:  # Last 3 files
            with open(json_file) as f:
                data = json.load(f)
                properties.extend(data.get('properties', []))
        
        # Filter by direct filters
        if parsed.direct_filters.max_price:
            properties = [p for p in properties if p.get('price', 0) <= parsed.direct_filters.max_price]
        
        console.print(f"  ✅ Loaded {len(properties)} properties from cache")
    
    if not properties:
        console.print("  ❌ No properties to analyze!", style="bold red")
        return
    
    # Limit for analysis
    properties_to_analyze = properties[:min(20, len(properties))]
    console.print(f"  📊 Analyzing top {len(properties_to_analyze)} properties")
    
    # ============================================================
    # STEP 3: Analyze Properties
    # ============================================================
    console.print(f"\n[bold cyan]STEP 3: Analyzing Properties (backend: {args.backend})[/bold cyan]")
    
    analyzer = PropertyTextAnalyzer(backend=args.backend)
    
    analyses = []
    with Progress() as progress:
        task = progress.add_task("[green]Analyzing...", total=len(properties_to_analyze))
        
        for prop in properties_to_analyze:
            # Convertir Property object a dict si es necesario
            if hasattr(prop, 'to_dict'):
                prop_dict = prop.to_dict()
            else:
                prop_dict = prop
            
            analysis = analyzer.analyze(
                property_id=prop_dict['id'],
                description=prop_dict.get('description', ''),
                generate_embedding=True
            )
            analyses.append(analysis)
            progress.update(task, advance=1)
    
    console.print(f"  ✅ Analyzed {len(analyses)} properties")
    
    # ============================================================
    # STEP 4: Match & Rank
    # ============================================================
    console.print("\n[bold cyan]STEP 4: Matching & Ranking[/bold cyan]")
    
    # Build requirements from indirect filters
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
    if parsed.indirect_filters.layout:
        requirements.append(QueryRequirement(
            feature_name=parsed.indirect_filters.layout.replace(' ', '_'),
            importance=0.7
        ))
    if parsed.indirect_filters.features:
        for feature in parsed.indirect_filters.features:
            requirements.append(QueryRequirement(
                feature_name=feature,
                importance=0.6
            ))
    
    # Match all properties
    matches = []
    for analysis in analyses:
        match = analyzer.match_against_query(
            analysis,
            query,
            requirements if requirements else None
        )
        matches.append((match, analysis))
    
    # Sort by score
    matches.sort(key=lambda x: x[0].final_score, reverse=True)
    
    console.print(f"  ✅ Ranked {len(matches)} properties")
    
    # ============================================================
    # STEP 5: Display Results
    # ============================================================
    console.print("\n" + "="*80)
    console.print(f"[bold green]TOP {args.max_results} MATCHES[/bold green]")
    console.print("="*80 + "\n")
    
    for i, (match, analysis) in enumerate(matches[:args.max_results], 1):
        # Encontrar property (puede ser objeto o dict)
        if hasattr(properties_to_analyze[0], 'to_dict'):
            prop = next(p.to_dict() for p in properties_to_analyze 
                    if p.id == match.property_id)
        else:
            prop = next(p for p in properties_to_analyze 
                    if p['id'] == match.property_id)
        
        # Quality indicator
        if match.final_score >= 0.7:
            indicator = "[bold green]⭐⭐⭐ EXCELLENT[/bold green]"
        elif match.final_score >= 0.5:
            indicator = "[yellow]⭐⭐ GOOD[/yellow]"
        elif match.final_score >= 0.3:
            indicator = "[orange3]⭐ FAIR[/orange3]"
        else:
            indicator = "[red]WEAK[/red]"
        
        console.print(f"[bold]#{i}. {prop.get('title', 'N/A')[:60]}...[/bold] {indicator}")
        console.print(f"   📍 {prop['location']}")
        console.print(f"   💰 {prop.get('price', 0):,}€ | {prop.get('size_m2', 'N/A')}m² | {prop.get('rooms', 'N/A')} hab")
        console.print(f"   🎯 Score: {match.final_score:.2f} (features: {match.feature_match_score:.2f}, semantic: {match.semantic_similarity_score:.2f})")
        
        # Top matched features
        if match.matched_features:
            console.print(f"   ✅ Matched: {', '.join(f.name for f in match.matched_features[:3])}")
        
        # Missing
        if match.missing_requirements:
            console.print(f"   ⚠️  Missing: {', '.join(match.missing_requirements)}")
        
        console.print(f"   🔗 {prop['url']}\n")
    
    # ============================================================
    # STEP 6: Summary Statistics
    # ============================================================
    console.print("\n" + "="*80)
    console.print("[bold cyan]SUMMARY[/bold cyan]")
    console.print("="*80 + "\n")
    
    scores = [m[0].final_score for m in matches]
    
    console.print(f"  Total properties analyzed: {len(matches)}")
    console.print(f"  Average score: {sum(scores)/len(scores):.2f}")
    console.print(f"\n  Score distribution:")
    console.print(f"    Excellent (≥0.7): {sum(1 for s in scores if s >= 0.7)} ({sum(1 for s in scores if s >= 0.7)/len(scores)*100:.0f}%)")
    console.print(f"    Good (≥0.5):      {sum(1 for s in scores if s >= 0.5)} ({sum(1 for s in scores if s >= 0.5)/len(scores)*100:.0f}%)")
    console.print(f"    Fair (≥0.3):      {sum(1 for s in scores if s >= 0.3)} ({sum(1 for s in scores if s >= 0.3)/len(scores)*100:.0f}%)")
    console.print(f"    Weak (<0.3):      {sum(1 for s in scores if s < 0.3)} ({sum(1 for s in scores if s < 0.3)/len(scores)*100:.0f}%)")
    
    # Features detected across all properties
    all_features = {}
    for analysis in analyses:
        for feat in analysis.detected_features:
            if feat.confidence >= 0.5:
                all_features[feat.name] = all_features.get(feat.name, 0) + 1
    
    if all_features:
        console.print(f"\n  Most common features detected:")
        for feat, count in sorted(all_features.items(), key=lambda x: x[1], reverse=True)[:10]:
            console.print(f"    • {feat}: {count} properties")
    
    console.print(f"\n✅ End-to-end test complete!")

if __name__ == "__main__":
    main()