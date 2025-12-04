"""
Run Hybrid Vision Analysis

Production-ready script con vision agent inteligente.
"""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.table import Table
import mlflow

from src.property_analysis.combined_analyzer import CombinedPropertyAnalyzer
from src.property_analysis.schemas import QueryRequirement

console = Console()


def load_properties(data_dir: str = "data/raw", limit: int = 20):
    """Load properties from scraped data"""
    data_path = Path(data_dir)
    properties = []
    
    for json_file in sorted(data_path.glob("fotocasa_*.json")):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                props = data.get('properties', [])
                properties.extend(props)
        except:
            continue
        
        if len(properties) >= limit * 2:
            break
    
    # Filter: only with images
    properties = [p for p in properties if p.get('images') and len(p['images']) > 0]
    return properties[:limit]


def main():
    console.print("[bold blue]🚀 Hybrid Vision Analysis[/bold blue]\n")
    
    # Configuration
    query = input("Query: ") or "Local comercial Barcelona con entrada independiente, luz natural"
    
    requirements = [
        QueryRequirement(feature_name="entrada_independiente", importance=1.0, required=True),
        QueryRequirement(feature_name="luz_natural", importance=0.8, required=False),
        QueryRequirement(feature_name="local_comercial", importance=0.9, required=False)
    ]
    
    # Vision mode (changeable)
    import platform
    is_mac = platform.system() == "Darwin"
    
    if is_mac:
        console.print("[yellow]Detected Mac - using qwen_only mode (no CUDA needed)[/yellow]")
        vision_mode = "qwen_only"
    else:
        console.print("[green]Detected Linux/Windows - using claude_primary mode[/green]")
        vision_mode = "claude_primary"
    
    # Load properties
    console.print(f"\n[cyan]Loading properties...[/cyan]")
    properties = load_properties(limit=20)
    console.print(f"✅ Loaded {len(properties)} properties\n")
    
    # Initialize analyzer
    analyzer = CombinedPropertyAnalyzer(
        vision_mode=vision_mode,
        vision_budget=100,
        qwen_confidence_threshold=0.7,
        enable_vision=True
    )
    
    # MLflow tracking
    mlflow.set_experiment("hybrid_vision_analysis")
    
    with mlflow.start_run():
        mlflow.log_param("query", query)
        mlflow.log_param("num_properties", len(properties))
        mlflow.log_param("vision_mode", vision_mode)
        
        # Analyze
        results = analyzer.analyze_batch_stage1(
            properties=properties,
            query_text=query,
            requirements=requirements,
            use_vision_agent=True
        )
        
        # Display results
        table = Table(title="Top 10 Properties")
        table.add_column("Rank", style="cyan")
        table.add_column("Property ID", style="yellow")
        table.add_column("Score", style="green")
        table.add_column("Vision", style="magenta")
        table.add_column("Price", style="blue")
        
        for i, r in enumerate(results[:10], 1):
            prop = r['property']
            vision_icon = "🔍" if r['needs_vision'] else "-"
            
            table.add_row(
                str(i),
                prop['id'][:15],
                f"{r['score']:.3f}",
                vision_icon,
                f"€{prop.get('price', 0):,}"
            )
        
        console.print("\n")
        console.print(table)
        
        # Status
        status = analyzer.vision_analyzer.get_budget_status()
        
        console.print("\n[bold]Analysis Summary:[/bold]")
        console.print(f"  Properties analyzed: {len(results)}")
        console.print(f"  Vision used: {sum(1 for r in results if r['needs_vision'])} properties")
        console.print(f"  Claude calls: {status['claude_calls_made']}")
        console.print(f"  Qwen calls: {status['qwen_calls_made']}")
        console.print(f"  Total cost: €{status['total_cost_eur']:.2f}")
        
        # Log metrics
        mlflow.log_metric("top_score", results[0]['score'])
        mlflow.log_metric("vision_used", sum(1 for r in results if r['needs_vision']))
        mlflow.log_metric("total_cost", status['total_cost_eur'])


if __name__ == "__main__":
    main()