"""
Test Qwen2-VL vs Claude Vision

Compara calidad, velocidad y costo de ambos modelos.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.property_analysis.vision_analyzer import VisionAnalyzer

console = Console()


def load_test_property(data_dir: str = "data/raw"):
    """Load one property for testing"""
    data_path = Path(data_dir)
    
    for json_file in sorted(data_path.glob("fotocasa_*.json")):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                props = data.get('properties', [])
                
                # Find property with images
                for p in props:
                    if p.get('images') and len(p['images']) >= 3:
                        return p
        except:
            continue
    
    return None


def test_mode(analyzer: VisionAnalyzer, property_data: dict, mode: str):
    """Test a specific mode"""
    console.print(f"\n[bold cyan]Testing: {mode.upper()}[/bold cyan]")
    
    start = time.time()
    
    try:
        description = analyzer.generate_photo_description(
            image_urls=property_data['images'],
            max_images=3,
            force_mode=mode
        )
        
        elapsed = time.time() - start
        
        # Get status
        status = analyzer.get_budget_status()
        
        console.print(f"✅ Success!")
        console.print(f"   Time: {elapsed:.2f}s")
        console.print(f"   Cost: €{status['total_cost_eur']:.3f}")
        console.print(f"   Length: {len(description)} chars")
        console.print(f"\n[bold]Description:[/bold]")
        console.print(description[:500] + "..." if len(description) > 500 else description)
        
        return {
            'success': True,
            'time': elapsed,
            'cost': status['total_cost_eur'],
            'description': description
        }
        
    except Exception as e:
        elapsed = time.time() - start
        console.print(f"[red]❌ Error: {e}[/red]")
        
        return {
            'success': False,
            'time': elapsed,
            'cost': 0,
            'error': str(e)
        }


def main():
    console.print(Panel.fit(
        "[bold blue]Qwen2-VL vs Claude Vision Comparison[/bold blue]",
        border_style="blue"
    ))
    
    # Load test property
    console.print("\n[cyan]Loading test property...[/cyan]")
    prop = load_test_property()
    
    if not prop:
        console.print("[red]No property found![/red]")
        return
    
    console.print(f"✅ Property: {prop['id']}")
    console.print(f"   Images: {len(prop['images'])}")
    console.print(f"   URL: {prop['url']}")
    
    # Test modes
    modes = ['qwen_only', 'claude_only', 'claude_primary']
    results = {}
    
    for mode in modes:
        analyzer = VisionAnalyzer(mode=mode)
        results[mode] = test_mode(analyzer, prop, mode)
    
    # Comparison table
    console.print("\n")
    console.print(Panel.fit(
        "[bold green]📊 Results Comparison[/bold green]",
        border_style="green"
    ))
    
    table = Table(title="Performance Comparison")
    table.add_column("Mode", style="cyan")
    table.add_column("Status", style="yellow")
    table.add_column("Time (s)", style="magenta")
    table.add_column("Cost (€)", style="green")
    table.add_column("Chars", style="blue")
    
    for mode, result in results.items():
        if result['success']:
            table.add_row(
                mode,
                "✅",
                f"{result['time']:.2f}",
                f"{result['cost']:.3f}",
                str(len(result['description']))
            )
        else:
            table.add_row(
                mode,
                "❌",
                f"{result['time']:.2f}",
                "-",
                "Error"
            )
    
    console.print(table)
    
    # Recommendations
    console.print("\n[bold]Recommendations:[/bold]")
    
    if results['qwen_only']['success']:
        qwen_time = results['qwen_only']['time']
        console.print(f"  • Qwen2-VL: {qwen_time:.1f}s per property, €0 cost")
        console.print(f"    → Good for: bulk processing, testing, development")
    
    if results['claude_only']['success']:
        claude_time = results['claude_only']['time']
        claude_cost = results['claude_only']['cost']
        console.print(f"  • Claude Vision: {claude_time:.1f}s per property, €{claude_cost:.3f} cost")
        console.print(f"    → Good for: production quality, critical cases")
    
    console.print(f"\n  • [bold]claude_primary[/bold] (recommended): Uses Claude by default, Qwen backup")
    console.print(f"    - Cost: €{results['claude_primary']['cost']:.3f} per property")
    console.print(f"    - Guarantees quality with budget protection")


if __name__ == "__main__":
    main()