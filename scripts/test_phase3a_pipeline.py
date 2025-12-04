"""
End-to-End Test: Phase 3A Complete Pipeline

Tests the full two-stage pipeline:
  Stage 1: Text + CLIP analysis (all properties, free)
  Stage 2: Claude Vision (top candidates, budget-conscious)

Compares with baseline (text-only) to validate improvement.
"""
import json
from pathlib import Path
from typing import List, Dict
import time
import mlflow
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.property_analysis.combined_analyzer import CombinedPropertyAnalyzer
from src.property_analysis.text_analyzer import PropertyTextAnalyzer
from src.property_analysis.schemas import QueryRequirement

console = Console()


def load_properties(data_dir: str = "data/raw", limit: int = 50) -> List[dict]:
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


def run_baseline_text_only(
    properties: List[dict],
    query: str,
    requirements: List[QueryRequirement]
) -> List[Dict]:
    """Baseline: Text-only analysis"""
    console.print("\n[bold yellow]📊 Baseline: Text-Only Analysis[/bold yellow]")
    
    text_analyzer = PropertyTextAnalyzer(backend="api")
    results = []
    
    for i, prop in enumerate(properties, 1):
        prop_id = prop.get('id', f'prop_{i}')
        
        # Text analysis
        analysis = text_analyzer.analyze(
            property_id=prop_id,
            description=prop.get('description', ''),
            generate_embedding=True
        )
        
        # Match
        match = text_analyzer.match_against_query(
            property_analysis=analysis,
            query_text=query,
            required_features=requirements,
            feature_weight=0.7,
            semantic_weight=0.3
        )
        
        results.append({
            'property_id': prop_id,
            'score': match.final_score,
            'matched_features': [f.name for f in match.matched_features],
            'url': prop.get('url', '')
        })
    
    results.sort(key=lambda x: x['score'], reverse=True)
    
    console.print(f"✅ Analyzed {len(properties)} properties")
    console.print(f"   Top score: {results[0]['score']:.3f}")
    console.print(f"   Avg score: {sum(r['score'] for r in results) / len(results):.3f}")
    
    return results


def run_stage1_text_clip(
    properties: List[dict],
    query: str,
    requirements: List[QueryRequirement],
    analyzer: CombinedPropertyAnalyzer
) -> List[Dict]:
    """Stage 1: Text + CLIP (free)"""
    console.print("\n[bold cyan]🚀 Stage 1: Text + CLIP Analysis (FREE)[/bold cyan]")
    
    start = time.time()
    
    results = analyzer.analyze_batch_stage1(
        properties=properties,
        query_text=query,
        requirements=requirements,
        log_to_mlflow=True
    )
    
    elapsed = time.time() - start
    
    console.print(f"✅ Analyzed {len(properties)} properties in {elapsed:.1f}s")
    console.print(f"   Top score: {results[0]['score']:.3f}")
    console.print(f"   Avg score: {sum(r['score'] for r in results) / len(results):.3f}")
    console.print(f"   Cost: €0 (CLIP is free!)")
    
    return results


def run_stage2_claude_vision(
    stage1_results: List[Dict],
    query: str,
    requirements: List[QueryRequirement],
    analyzer: CombinedPropertyAnalyzer,
    top_n: int = 20
) -> List[Dict]:
    """Stage 2: Claude Vision on top candidates"""
    console.print(f"\n[bold green]🔍 Stage 2: Claude Vision on Top {top_n} (BUDGET-AWARE)[/bold green]")
    
    start = time.time()
    
    results = analyzer.enhance_top_candidates_stage2(
        candidates=stage1_results,
        query_text=query,
        requirements=requirements,
        top_n=top_n,
        log_to_mlflow=True
    )
    
    elapsed = time.time() - start
    budget = analyzer.vision_analyzer.get_budget_status()
    
    console.print(f"✅ Enhanced top {min(top_n, len(stage1_results))} properties in {elapsed:.1f}s")
    console.print(f"   Top score: {results[0]['score']:.3f}")
    console.print(f"   Avg score: {sum(r['score'] for r in results) / len(results):.3f}")
    console.print(f"   Cost: €{budget['total_cost_eur']:.2f}")
    console.print(f"   Remaining budget: €{budget['remaining_budget_eur']:.2f}")
    
    return results


def compare_results(
    baseline: List[Dict],
    stage1: List[Dict],
    stage2: List[Dict]
):
    """Compare all three approaches"""
    console.print("\n")
    console.print(Panel.fit(
        "[bold blue]📈 Results Comparison[/bold blue]",
        border_style="blue"
    ))
    
    # Top 10 comparison
    table = Table(title="Top 10 Properties by Score")
    table.add_column("Rank", style="cyan", width=6)
    table.add_column("Text Only", style="yellow", width=25)
    table.add_column("Score", style="yellow", width=8)
    table.add_column("+ CLIP", style="cyan", width=25)
    table.add_column("Score", style="cyan", width=8)
    table.add_column("+ Vision", style="green", width=25)
    table.add_column("Score", style="green", width=8)
    
    for i in range(min(10, len(baseline))):
        table.add_row(
            str(i + 1),
            baseline[i]['property_id'][:25],
            f"{baseline[i]['score']:.3f}",
            stage1[i]['property']['id'][:25],
            f"{stage1[i]['score']:.3f}",
            stage2[i]['property']['id'][:25],
            f"{stage2[i]['score']:.3f}"
        )
    
    console.print(table)
    
    # Statistics
    baseline_top = baseline[0]['score']
    stage1_top = stage1[0]['score']
    stage2_top = stage2[0]['score']
    
    stage1_improvement = (stage1_top - baseline_top) * 100
    stage2_improvement = (stage2_top - baseline_top) * 100
    
    console.print("\n[bold]Summary Statistics:[/bold]")
    console.print(f"  Text Only (baseline):  {baseline_top:.3f}")
    console.print(f"  + CLIP (Stage 1):      {stage1_top:.3f}  ({stage1_improvement:+.1f}%)")
    console.print(f"  + Vision (Stage 2):    {stage2_top:.3f}  ({stage2_improvement:+.1f}%)")
    
    # Feature detection
    console.print("\n[bold]Feature Detection:[/bold]")
    
    # Count "entrada_independiente" detections
    baseline_entrada = sum(1 for r in baseline if 'entrada_independiente' in r['matched_features'])
    stage1_entrada = sum(1 for r in stage1 if any(
        'entrada' in f.name.lower() or 'independiente' in f.name.lower()
        for f in r['text_analysis'].detected_features
    ))
    stage2_entrada = sum(1 for r in stage2[:20] if (  # Only top 20 had Stage 2
        any('entrada' in f.name.lower() or 'independiente' in f.name.lower()
           for f in r['text_analysis'].detected_features) or
        (r.get('vision_analysis') and any(
            'entrada' in f.name.lower() or 'independiente' in f.name.lower()
            for f in r['vision_analysis'].detected_features
        ))
    ))
    
    console.print(f"  'entrada_independiente' detected:")
    console.print(f"    Text only: {baseline_entrada}/{len(baseline)} ({baseline_entrada/len(baseline)*100:.1f}%)")
    console.print(f"    + CLIP:    {stage1_entrada}/{len(stage1)} ({stage1_entrada/len(stage1)*100:.1f}%)")
    console.print(f"    + Vision:  {stage2_entrada}/20 (top 20: {stage2_entrada/20*100:.1f}%)")
    
    return {
        'baseline_top': baseline_top,
        'stage1_top': stage1_top,
        'stage2_top': stage2_top,
        'stage1_improvement': stage1_improvement,
        'stage2_improvement': stage2_improvement
    }


def display_top_match_details(results: List[Dict], stage: str):
    """Show detailed analysis of top match"""
    console.print(f"\n[bold]🎯 Top Match Details ({stage}):[/bold]")
    
    top = results[0]
    prop = top['property']
    match = top['match']
    
    console.print(f"  Property: {prop.get('id')}")
    console.print(f"  Score: {match.final_score:.3f}")
    console.print(f"  URL: {prop.get('url', 'N/A')[:80]}")
    
    console.print(f"\n  Matched Features ({len(match.matched_features)}):")
    for f in match.matched_features[:10]:  # Top 10
        console.print(f"    • {f.name}: {f.confidence:.2f} ({f.source})")
    
    if match.missing_requirements:
        console.print(f"\n  Missing Requirements ({len(match.missing_requirements)}):")
        for m in match.missing_requirements[:5]:
            console.print(f"    • {m}")


def main():
    """Run complete Phase 3A test"""
    console.print(Panel.fit(
        "[bold blue]Phase 3A: Complete Pipeline Test[/bold blue]\n"
        "Testing: Text-Only vs Text+CLIP vs Text+CLIP+Vision",
        border_style="blue"
    ))
    
    # Configuration
    query = "Local comercial Barcelona con entrada independiente, luz natural, máximo 300 mil euros"
    requirements = [
        QueryRequirement(feature_name="entrada_independiente", importance=1.0, required=True),
        QueryRequirement(feature_name="luz_natural", importance=0.8, required=False),
        QueryRequirement(feature_name="local_comercial", importance=0.9, required=False)
    ]
    
    num_properties = 20  # Reasonable test size
    top_n_stage2 = 10  # Claude Vision budget control (~€0.50)
    
    console.print(f"\n[bold]Configuration:[/bold]")
    console.print(f"  Query: {query}")
    console.print(f"  Requirements: {len(requirements)} features")
    console.print(f"  Properties: {num_properties}")
    console.print(f"  Stage 2 candidates: {top_n_stage2}")
    console.print(f"  Expected cost: ~€{top_n_stage2 * 0.05:.2f}")
    
    # Load data
    console.print("\n[cyan]Loading properties...[/cyan]")
    properties = load_properties(limit=num_properties)
    console.print(f"✅ Loaded {len(properties)} properties with images\n")
    
    # MLflow tracking
    mlflow.set_experiment("phase3a_complete_pipeline")
    
    with mlflow.start_run(run_name="phase3a_full_test"):
        mlflow.log_param("query", query)
        mlflow.log_param("num_properties", len(properties))
        mlflow.log_param("stage2_candidates", top_n_stage2)
        
        # Baseline: Text only
        baseline_results = run_baseline_text_only(properties, query, requirements)
        
        # Initialize combined analyzer
        analyzer = CombinedPropertyAnalyzer(
            text_backend="api",
            enable_vision=True,
            vision_budget=300  # Max budget
        )
        
        # Stage 1: Text + CLIP
        stage1_results = run_stage1_text_clip(properties, query, requirements, analyzer)
        
        # Stage 2: Claude Vision on top candidates
        stage2_results = run_stage2_claude_vision(
            stage1_results, query, requirements, analyzer, top_n=top_n_stage2
        )
        
        # Compare
        stats = compare_results(baseline_results, stage1_results, stage2_results)
        
        # Log to MLflow
        mlflow.log_metric("baseline_top_score", stats['baseline_top'])
        mlflow.log_metric("stage1_top_score", stats['stage1_top'])
        mlflow.log_metric("stage2_top_score", stats['stage2_top'])
        mlflow.log_metric("stage1_improvement_percent", stats['stage1_improvement'])
        mlflow.log_metric("stage2_improvement_percent", stats['stage2_improvement'])
        
        # Show details
        display_top_match_details(stage2_results, "Stage 2: Full Vision")
        
        # Budget status
        budget = analyzer.get_status()
        console.print(f"\n[bold]Budget Status:[/bold]")
        if budget['vision_enabled']:
            console.print(f"  Calls made: {budget['vision_budget']['calls_made']}")
            console.print(f"  Remaining: {budget['vision_budget']['remaining']}")
            console.print(f"  Total cost: €{budget['vision_budget']['total_cost_eur']:.2f}")
        
        # Save results
        results_dir = Path("data/experiments/phase3a_pipeline")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        with open(results_dir / "baseline.json", 'w') as f:
            json.dump(baseline_results, f, indent=2)
        
        console.print(f"\n✅ Results saved to {results_dir}")
        console.print(f"✅ MLflow: http://localhost:5000")
        
        # Decision
        console.print("\n[bold green]📋 Conclusions:[/bold green]")
        
        if stats['stage1_improvement'] > 10:
            console.print("  ✅ CLIP adds significant value (+10% improvement)")
            console.print("  ✅ Stage 1 should be used for all properties")
        else:
            console.print("  ⚠️  CLIP improvement is modest (<10%)")
        
        if stats['stage2_improvement'] > 15:
            console.print("  ✅ Claude Vision adds major value (+15% improvement)")
            console.print("  ✅ Stage 2 justified for top candidates")
        elif stats['stage2_improvement'] > stats['stage1_improvement'] + 5:
            console.print("  ✅ Claude Vision provides additional boost")
            console.print("  📝 Consider expanding Stage 2 to more candidates")
        else:
            console.print("  ⚠️  Claude Vision improvement is marginal")
            console.print("  💡 Consider: better prompts or focus on text analysis")
        
        console.print(f"\n  Budget used: €{budget['vision_budget']['total_cost_eur']:.2f} / €20")
        console.print(f"  Cost per property (Stage 2): €{budget['vision_budget']['total_cost_eur'] / top_n_stage2:.3f}")


if __name__ == "__main__":
    main()