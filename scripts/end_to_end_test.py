#!/usr/bin/env python3
"""
Test end-to-end completo: Query → Scrape → Analyze → Match

Uso:
    python scripts/end_to_end_test.py "Local Barcelona entrada independiente máx 250k"
    python scripts/end_to_end_test.py "Piso luminoso 3 habitaciones" --skip-scrape
    python scripts/end_to_end_test.py "Local entrada independiente" --vision-mode claude_primary
    python scripts/end_to_end_test.py "Piso elegante acogedor" --vision-mode qwen_only --use-vision-agent

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
import platform
import torch

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.query_parser.parser import QueryParser
from src.data.scraper import FotocasaScraper
from src.property_analysis.combined_analyzer import CombinedPropertyAnalyzer
from src.property_analysis.schemas import QueryRequirement
from src.data_quality.validator import FotocasaValidator

console = Console()

def get_default_vision_mode():
    """Auto-detect best vision mode for current hardware"""
    system = platform.system()
    
    # Mac with Apple Silicon
    if system == "Darwin":
        if torch.backends.mps.is_available():
            return "qwen_only", "Mac M3 detected - using MPS acceleration"
        else:
            return "claude_only", "Mac without MPS - using Claude API only"
    
    # Linux/Windows with GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        if "RTX" in gpu_name or "A100" in gpu_name:
            return "claude_primary", f"Powerful GPU detected ({gpu_name})"
        else:
            return "qwen_primary", f"GPU detected ({gpu_name})"
    
    # CPU only
    return "claude_only", "No GPU detected - using Claude API only"

def parse_arguments():
    parser = argparse.ArgumentParser(description="End-to-end property search test")
    parser.add_argument("query", nargs="+", help="Search query in Spanish")
    parser.add_argument("--skip-scrape", action="store_true", 
                       help="Skip scraping, use existing data")
    parser.add_argument("--backend", choices=["api", "local"], default="api",
                       help="Text analyzer backend (default: api)")
    parser.add_argument("--vision-mode", 
                       choices=["claude_primary", "qwen_primary", "claude_only", "qwen_only", "auto"],
                       default="auto",
                       help="Vision analysis mode (default: auto-detect)")
    parser.add_argument("--use-vision-agent", action="store_true",
                        help="Use LLM agent to decide when to use vision (recommended)")
    parser.add_argument("--vision-budget", type=int, default=100,
                        help="Max Claude vision calls (default: 100)")
    parser.add_argument("--no-vision", action="store_true",
                       help="Disable vision analysis completely")
    parser.add_argument("--max-results", type=int, default=10,
                       help="Max properties to show (default: 10)")
    parser.add_argument("--verbose", action="store_true",
                       help="Show detailed vision descriptions")
    return parser.parse_args()

def save_and_display_vision_details(result: dict, output_dir: Path, args):
    """Save images and display vision description in verbose mode"""
    if not args.verbose or not result.get('needs_vision'):
        return
    
    prop = result['property']
    prop_id = prop['id']
    
    # Create property debug dir
    prop_dir = output_dir / prop_id
    prop_dir.mkdir(parents=True, exist_ok=True)
    
    # Save first 3 images
    import requests
    from PIL import Image
    from io import BytesIO
    
    images_saved = []
    for idx, url in enumerate(prop.get('images', [])[:3], 1):
        try:
            response = requests.get(url, timeout=10)
            img = Image.open(BytesIO(response.content))
            filepath = prop_dir / f"image_{idx}.jpg"
            img.save(filepath)
            images_saved.append(str(filepath))
        except:
            continue
    
    # Extract vision description from analysis
    analysis = result.get('text_analysis')
    if analysis and hasattr(analysis, 'original_text'):
        full_text = analysis.original_text or ""
        
        # Find vision part (starts with [Análisis de or [Foto)
        if "[Análisis de" in full_text or "[Foto" in full_text:
            parts = full_text.split("\n\n")
            vision_parts = [p for p in parts if p.startswith("[Análisis de") or p.startswith("[Foto")]
            
            if vision_parts:
                return {
                    'images': images_saved,
                    'description': "\n".join(vision_parts),
                    'path': str(prop_dir)
                }
    
    return None

def main():
    args = parse_arguments()
    query = " ".join(args.query)
    
    console.print(Panel.fit(f"🚀 END-TO-END TEST\nQuery: {query}", style="bold blue"))

    # Determine vision mode
    if args.no_vision:
        vision_mode = None
        vision_enabled = False
        console.print("[yellow]Vision analysis: DISABLED[/yellow]")
    else:
        if args.vision_mode == "auto":
            vision_mode, reason = get_default_vision_mode()
            console.print(f"[cyan]Vision mode: {vision_mode} (auto-detected: {reason})[/cyan]")
        else:
            vision_mode = args.vision_mode
            console.print(f"[cyan]Vision mode: {vision_mode}[/cyan]")
        
        vision_enabled = True
        if args.use_vision_agent:
            console.print("[cyan]Vision agent: ENABLED (LLM decides when to use vision)[/cyan]")
        else:
            console.print("[yellow]Vision agent: DISABLED (will analyze all properties)[/yellow]")

    
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
        "max_pages": 1
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
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    properties.extend(data.get('properties', []))
            except json.JSONDecodeError:
                console.print(f"  ⚠️  Skipping corrupted file: {json_file.name}")
                continue
        
        # Filter by direct filters
        if parsed.direct_filters.max_price:
            properties = [p for p in properties if p.get('price', 0) <= parsed.direct_filters.max_price]
        
        console.print(f"  ✅ Loaded {len(properties)} properties from cache")
    
    if not properties:
        console.print("  ❌ No properties to analyze!", style="bold red")
        return
    
    # Limit for analysis
    properties_to_analyze = properties[:min(5, len(properties))]
    console.print(f"  📊 Analyzing top {len(properties_to_analyze)} properties")
    
    # ============================================================
    # STEP 3: Analyze Properties
    # ============================================================
    console.print(f"\n[bold cyan]STEP 3: Analyzing Properties[/bold cyan]")
    console.print(f"  Text backend: {args.backend}")
    if vision_enabled:
        console.print(f"  Vision mode: {vision_mode}")
        console.print(f"  Vision agent: {'enabled' if args.use_vision_agent else 'disabled'}")

    # Initialize analyzer (with or without vision)
    analyzer = CombinedPropertyAnalyzer(
        text_backend=args.backend,
        enable_vision=vision_enabled,
        vision_mode=vision_mode if vision_enabled else None,
        vision_budget=args.vision_budget,
        qwen_confidence_threshold=0.7
    )
    
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
    
    # Convert properties to dicts if needed
    props_list = []
    for prop in properties_to_analyze:
        if hasattr(prop, 'to_dict'):
            props_list.append(prop.to_dict())
        else:
            props_list.append(prop)
    
    # Analyze with intelligent vision usage
    results = analyzer.analyze_batch_stage1(
        properties=props_list,
        query_text=query,
        requirements=requirements if requirements else [],
        use_vision_agent=args.use_vision_agent,
        log_to_mlflow=False
    )

    # Validate results
    if results is None or len(results) == 0:
        console.print("[red]  ❌ Analysis failed - no results returned[/red]")
        console.print("\n[yellow]Debugging info:[/yellow]")
        console.print(f"  Properties to analyze: {len(props_list)}")
        console.print(f"  Query: {query}")
        console.print(f"  Requirements: {len(requirements)}")
        
        # Try to get more info
        if hasattr(analyzer, 'vision_analyzer') and analyzer.vision_analyzer:
            try:
                status = analyzer.vision_analyzer.get_budget_status()
                console.print(f"  Vision status: {status}")
            except:
                pass
        
        return

    console.print(f"  ✅ Analyzed {len(results)} properties")
    # Get vision statistics
    if vision_enabled:
        try:
            vision_status = analyzer.vision_analyzer.get_budget_status()
            console.print(f"\n  Vision Statistics:")
            console.print(f"    Properties with vision: {sum(1 for r in results if r['needs_vision'])}")
            console.print(f"    Claude calls: {vision_status['claude_calls_made']}")
            console.print(f"    Qwen calls: {vision_status['qwen_calls_made']}")
            console.print(f"    Total cost: €{vision_status['total_cost_eur']:.2f}")
        except Exception as e:
            console.print(f"[yellow]  ⚠️  Could not get vision statistics: {e}[/yellow]")
    
    # ============================================================
    # STEP 4: Display Results
    # ============================================================
    console.print("\n" + "="*80)
    console.print(f"[bold green]TOP {args.max_results} MATCHES[/bold green]")
    console.print("="*80 + "\n")
    
    for i, result in enumerate(results[:args.max_results], 1):
        prop = result['property']
        match = result['text_match']
        used_vision = result['needs_vision']
      

        # Quality indicator
        if match.final_score >= 0.7:
            indicator = "[bold green]⭐⭐⭐ EXCELLENT[/bold green]"
        elif match.final_score >= 0.5:
            indicator = "[yellow]⭐⭐ GOOD[/yellow]"
        elif match.final_score >= 0.3:
            indicator = "[orange3]⭐ FAIR[/orange3]"
        else:
            indicator = "[red]WEAK[/red]"

        vision_icon = " 🔍" if used_vision else ""
        
        console.print(f"[bold]#{i}. {prop.get('title', 'N/A')[:60]}...[/bold] {indicator}{vision_icon}")
        console.print(f"   📍 {prop['location']}")
        console.print(f"   💰 {prop.get('price', 0):,}€ | {prop.get('size_m2', 'N/A')}m² | {prop.get('rooms', 'N/A')} hab")
        console.print(f"   🎯 Score: {match.final_score:.2f} (features: {match.feature_match_score:.2f}, semantic: {match.semantic_similarity_score:.2f})")
        
        # Top matched features
        if match.matched_features:
            console.print(f"   ✅ Matched: {', '.join(f.name for f in match.matched_features[:3])}")
        
        # Missing
        if match.missing_requirements:
            console.print(f"   ⚠️  Missing: {', '.join(match.missing_requirements)}")

        # Verbose: Show vision description
        if args.verbose and used_vision:
            # Save images and get description
           from pathlib import Path
           output_dir = Path("data/debug/end_to_end")
           vision_details = save_and_display_vision_details(result, output_dir, args)
          
           if vision_details:
               console.print(f"\n   [bold cyan]🔍 Vision Analysis:[/bold cyan]")
               console.print(f"   {vision_details['description'][:300]}...")
               console.print(f"\n   [bold]📸 Images saved:[/bold] {vision_details['path']}")
               console.print(f"   open {vision_details['path']}")
        
        
        console.print(f"   🔗 {prop['url']}\n")
    
    # ============================================================
    # STEP 5: Summary Statistics
    # ============================================================
    console.print("\n" + "="*80)
    console.print("[bold cyan]SUMMARY[/bold cyan]")
    console.print("="*80 + "\n")
    
    scores = [r['score'] for r in results]
    
    console.print(f"  Total properties analyzed: {len(results)}")
    console.print(f"  Average score: {sum(scores)/len(scores):.2f}")
    console.print(f"\n  Score distribution:")
    console.print(f"    Excellent (≥0.7): {sum(1 for s in scores if s >= 0.7)} ({sum(1 for s in scores if s >= 0.7)/len(scores)*100:.0f}%)")
    console.print(f"    Good (≥0.5):      {sum(1 for s in scores if s >= 0.5)} ({sum(1 for s in scores if s >= 0.5)/len(scores)*100:.0f}%)")
    console.print(f"    Fair (≥0.3):      {sum(1 for s in scores if s >= 0.3)} ({sum(1 for s in scores if s >= 0.3)/len(scores)*100:.0f}%)")
    console.print(f"    Weak (<0.3):      {sum(1 for s in scores if s < 0.3)} ({sum(1 for s in scores if s < 0.3)/len(scores)*100:.0f}%)")
    
        # Vision usage stats
    if vision_enabled:
        vision_used = sum(1 for r in results if r['needs_vision'])
        console.print(f"\n  Vision analysis:")
        console.print(f"    Used on: {vision_used}/{len(results)} properties ({vision_used/len(results)*100:.0f}%)")
        if vision_status['claude_calls_made'] > 0:
            console.print(f"    Claude calls: {vision_status['claude_calls_made']}")
        if vision_status['qwen_calls_made'] > 0:
            console.print(f"    Qwen calls: {vision_status['qwen_calls_made']}")
        console.print(f"    Total cost: €{vision_status['total_cost_eur']:.2f}")
   
    # Features detected across all properties
    all_features = {}
    for result in results:
        analysis = result['text_analysis']
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