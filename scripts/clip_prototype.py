"""
Quick Win Prototype: Test if CLIP embeddings improve property matching

Goal: Validate that image analysis adds value BEFORE investing in Claude Vision

Test:
1. Take 20 properties from your existing data
2. Analyze with text only (baseline)
3. Analyze with text + CLIP image embeddings
4. Compare top match scores

Cost: €0 (CLIP is free on M3)
Time: ~5 minutes for 20 properties
"""
import json
from pathlib import Path
from typing import List, Dict
import numpy as np
from rich.console import Console
from rich.table import Table
import mlflow

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.property_analysis.text_analyzer import PropertyTextAnalyzer
from src.property_analysis.vision_analyzer import VisionAnalyzer
from src.property_analysis.schemas import QueryRequirement

console = Console()


def load_properties(data_dir: str = "data/raw", limit: int = 20) -> List[dict]:
    """Load properties from scraped data"""
    data_path = Path(data_dir)
    properties = []
    
    for json_file in sorted(data_path.glob("fotocasa_*.json")):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                props = data.get('properties', [])
                properties.extend(props)
                console.print(f"  Loaded {len(props)} properties from {json_file.name}")
        except json.JSONDecodeError as e:
            console.print(f"  [yellow]⚠️  Skipping {json_file.name}: Invalid JSON at line {e.lineno}[/yellow]")
            continue
        except Exception as e:
            console.print(f"  [yellow]⚠️  Skipping {json_file.name}: {str(e)}[/yellow]")
            continue
        
        # Stop if we have enough
        if len(properties) >= limit * 2:  # Load extra for filtering
            break
    
    # Filter: only properties with images
    properties = [p for p in properties if p.get('images') and len(p['images']) > 0]
    
    if len(properties) == 0:
        console.print("[red]❌ No valid properties found with images![/red]")
        console.print(f"[yellow]Check that {data_dir} contains fotocasa_*.json files[/yellow]")
    
    return properties[:limit]


def analyze_text_only(
    properties: List[dict],
    query: str,
    requirements: List[QueryRequirement]
) -> List[Dict]:
    """Baseline: text analysis only"""
    console.print("\n[bold cyan]Stage 1: Text-Only Analysis[/bold cyan]")
    
    text_analyzer = PropertyTextAnalyzer(backend="api")
    results = []
    
    for i, prop in enumerate(properties, 1):
        prop_id = prop.get('id', f'prop_{i}')
        description = prop.get('description', '')
        
        console.print(f"  Analyzing {i}/{len(properties)}: {prop_id}")
        
        # Text analysis
        analysis = text_analyzer.analyze(
            property_id=prop_id,
            description=description,
            generate_embedding=True
        )
        
        # Match against query
        match = text_analyzer.match_against_query(
            property_analysis=analysis,
            query_text=query,
            required_features=requirements,
            feature_weight=0.7,
            semantic_weight=0.3
        )
        
        results.append({
            'property_id': prop_id,
            'text_score': match.final_score,
            'feature_score': match.feature_match_score,
            'semantic_score': match.semantic_similarity_score,
            'matched_features': [f.name for f in match.matched_features],
            'url': prop.get('url', '')
        })
    
    return results


def analyze_text_plus_vision(
    properties: List[dict],
    query: str,
    requirements: List[QueryRequirement]
) -> List[Dict]:
    """Enhanced: text + CLIP vision"""
    console.print("\n[bold cyan]Stage 2: Text + CLIP Vision Analysis[/bold cyan]")
    
    text_analyzer = PropertyTextAnalyzer(backend="api")
    vision_analyzer = VisionAnalyzer()
    
    results = []
    
    for i, prop in enumerate(properties, 1):
        prop_id = prop.get('id', f'prop_{i}')
        description = prop.get('description', '')
        images = prop.get('images', [])
        
        console.print(f"  Analyzing {i}/{len(properties)}: {prop_id}")
        
        # Text analysis
        text_analysis = text_analyzer.analyze(
            property_id=prop_id,
            description=description,
            generate_embedding=True
        )
        
        # Vision analysis (Stage 1: CLIP only)
        vision_analysis = vision_analyzer.analyze_property_stage1(
            property_id=prop_id,
            image_urls=images,
            max_images=3  # Keep it fast
        )
        
        # Compute image-text similarity
        avg_image_embedding = vision_analysis.get_avg_embedding()
        vision_score = 0.0
        
        if avg_image_embedding is not None:
            vision_score = vision_analyzer.compute_image_text_similarity(
                avg_image_embedding,
                query
            )
        
        # Get text matching scores
        text_match = text_analyzer.match_against_query(
            property_analysis=text_analysis,
            query_text=query,
            required_features=requirements,
            feature_weight=0.7,
            semantic_weight=0.3
        )
        
        # Combined score: 40% text features + 30% text semantic + 30% vision
        combined_score = (
            text_match.feature_match_score * 0.4 +
            text_match.semantic_similarity_score * 0.3 +
            vision_score * 0.3
        )
        
        results.append({
            'property_id': prop_id,
            'combined_score': combined_score,
            'text_feature_score': text_match.feature_match_score,
            'text_semantic_score': text_match.semantic_similarity_score,
            'vision_score': vision_score,
            'matched_features': [f.name for f in text_match.matched_features],
            'url': prop.get('url', '')
        })
    
    return results


def compare_results(text_results: List[Dict], vision_results: List[Dict]):
    """Compare and display results"""
    console.print("\n[bold green]📊 Results Comparison[/bold green]\n")
    
    # Sort by score
    text_sorted = sorted(text_results, key=lambda x: x['text_score'], reverse=True)
    vision_sorted = sorted(vision_results, key=lambda x: x['combined_score'], reverse=True)
    
    # Top 5 comparison
    table = Table(title="Top 5 Properties Comparison")
    table.add_column("Rank", style="cyan")
    table.add_column("Text Only", style="yellow")
    table.add_column("Score", style="yellow")
    table.add_column("Text + Vision", style="green")
    table.add_column("Score", style="green")
    table.add_column("Improvement", style="magenta")
    
    for i in range(min(5, len(text_sorted))):
        text_prop = text_sorted[i]
        vision_prop = vision_sorted[i]
        
        improvement = (vision_prop['combined_score'] - text_prop['text_score']) * 100
        improvement_str = f"+{improvement:.1f}%" if improvement > 0 else f"{improvement:.1f}%"
        
        table.add_row(
            str(i + 1),
            text_prop['property_id'][:20],
            f"{text_prop['text_score']:.3f}",
            vision_prop['property_id'][:20],
            f"{vision_prop['combined_score']:.3f}",
            improvement_str
        )
    
    console.print(table)
    
    # Statistics
    console.print("\n[bold]Statistics:[/bold]")
    console.print(f"  Text-only top score: {text_sorted[0]['text_score']:.3f}")
    console.print(f"  Text+Vision top score: {vision_sorted[0]['combined_score']:.3f}")
    console.print(f"  Improvement: {(vision_sorted[0]['combined_score'] - text_sorted[0]['text_score']) * 100:+.1f}%")
    
    # Features detected
    text_features = set()
    for r in text_results:
        text_features.update(r['matched_features'])
    
    console.print(f"\n  Unique features detected (text): {len(text_features)}")
    console.print(f"  Top features: {', '.join(list(text_features)[:5])}")
    
    return {
        'text_top_score': text_sorted[0]['text_score'],
        'vision_top_score': vision_sorted[0]['combined_score'],
        'improvement_percent': (vision_sorted[0]['combined_score'] - text_sorted[0]['text_score']) * 100
    }


def main():
    """Run quick win prototype"""
    console.print("[bold blue]🚀 Quick Win Prototype: CLIP Vision Test[/bold blue]\n")
    
    # Configuration
    query = "Local comercial Barcelona con entrada independiente, luz natural, máximo 300 mil euros"
    requirements = [
        QueryRequirement(feature_name="entrada_independiente", importance=1.0, required=True),
        QueryRequirement(feature_name="luz_natural", importance=0.8, required=False),
        QueryRequirement(feature_name="local_comercial", importance=0.9, required=False)
    ]
    
    console.print(f"[bold]Query:[/bold] {query}")
    console.print(f"[bold]Requirements:[/bold] {len(requirements)} features")
    console.print(f"[bold]Test size:[/bold] 10 properties (quick validation)\n")
    
    # Load data
    console.print("[cyan]Loading properties...[/cyan]")
    properties = load_properties(limit=10)
    console.print(f"✅ Loaded {len(properties)} properties with images\n")
    
    # MLflow tracking
    mlflow.set_experiment("phase3a_clip_prototype")
    
    with mlflow.start_run(run_name="clip_quick_win"):
        mlflow.log_param("query", query)
        mlflow.log_param("num_properties", len(properties))
        mlflow.log_param("test_type", "text_vs_text+clip")
        
        # Test 1: Text only
        text_results = analyze_text_only(properties, query, requirements)
        
        # Test 2: Text + CLIP Vision
        vision_results = analyze_text_plus_vision(properties, query, requirements)
        
        # Compare
        stats = compare_results(text_results, vision_results)
        
        # Log to MLflow
        mlflow.log_metric("text_only_top_score", stats['text_top_score'])
        mlflow.log_metric("text_vision_top_score", stats['vision_top_score'])
        mlflow.log_metric("improvement_percent", stats['improvement_percent'])
        
        # Save results
        results_dir = Path("data/experiments/clip_prototype")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        with open(results_dir / "text_only.json", 'w') as f:
            json.dump(text_results, f, indent=2)
        
        with open(results_dir / "text_vision.json", 'w') as f:
            json.dump(vision_results, f, indent=2)
        
        console.print(f"\n✅ Results saved to {results_dir}")
        console.print(f"✅ Cost: €0 (CLIP is free!)")
        console.print(f"\n[bold green]Next steps:[/bold green]")
        
        if stats['improvement_percent'] > 10:
            console.print("  ✅ Vision adds significant value! Proceed with full implementation")
            console.print("  📝 Consider adding Claude Vision for top candidates")
        elif stats['improvement_percent'] > 5:
            console.print("  ⚠️  Modest improvement. Test with more properties or refine weighting")
        else:
            console.print("  ❌ Minimal improvement. Consider focusing on better text analysis first")


if __name__ == "__main__":
    main()