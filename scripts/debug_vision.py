"""
Debug Vision Analysis - See what's happening!

Shows:
- Images being analyzed
- CLIP similarity scores per image
- Feature detection
- Visual output
"""
import json
from pathlib import Path
from typing import List
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import requests
from PIL import Image
from io import BytesIO

from src.property_analysis.text_analyzer import PropertyTextAnalyzer
from src.property_analysis.vision_analyzer import VisionAnalyzer
from src.property_analysis.schemas import QueryRequirement

console = Console()


def load_one_property(property_id: str = None, data_dir: str = "data/raw"):
    """Load a single property for detailed analysis"""
    data_path = Path(data_dir)
    
    for json_file in sorted(data_path.glob("fotocasa_*.json")):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                props = data.get('properties', [])
                
                if property_id:
                    # Find specific property
                    for p in props:
                        if p.get('id') == property_id:
                            return p
                else:
                    # Return first property with images
                    for p in props:
                        if p.get('images') and len(p['images']) > 0:
                            return p
        except:
            continue
    
    return None


def save_image(url: str, filepath: Path):
    """Download and save image"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        img.save(filepath)
        return True
    except Exception as e:
        console.print(f"[red]Failed to save {url}: {e}[/red]")
        return False


def analyze_property_debug(
    property_data: dict,
    query: str,
    requirements: List[QueryRequirement],
    save_images: bool = True
):
    """Detailed analysis with full transparency"""
    
    prop_id = property_data.get('id')
    console.print(Panel.fit(
        f"[bold blue]Analyzing Property: {prop_id}[/bold blue]",
        border_style="blue"
    ))
    
    # Create output directory
    output_dir = Path(f"data/debug/{prop_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Property info
    console.print("\n[bold]Property Details:[/bold]")
    console.print(f"  ID: {prop_id}")
    console.print(f"  Price: €{property_data.get('price', 'N/A'):,}")
    console.print(f"  Location: {property_data.get('location', 'N/A')}")
    console.print(f"  Size: {property_data.get('size_m2', 'N/A')}m²")
    console.print(f"  Images: {len(property_data.get('images', []))}")
    console.print(f"  URL: {property_data.get('url', 'N/A')}")
    
    # Description
    description = property_data.get('description', '')
    console.print(f"\n[bold]Description:[/bold]")
    console.print(f"  {description[:200]}..." if len(description) > 200 else f"  {description}")
    
    # ============= TEXT ANALYSIS =============
    console.print("\n[bold cyan]═══ TEXT ANALYSIS ═══[/bold cyan]")
    
    text_analyzer = PropertyTextAnalyzer(backend="api")
    text_analysis = text_analyzer.analyze(
        property_id=prop_id,
        description=description,
        generate_embedding=True
    )
    
    console.print(f"Features detected: {len(text_analysis.detected_features)}")
    
    if text_analysis.detected_features:
        table = Table(title="Text Features")
        table.add_column("Feature", style="cyan")
        table.add_column("Confidence", style="yellow")
        table.add_column("Value", style="green")
        
        for f in sorted(text_analysis.detected_features, key=lambda x: x.confidence, reverse=True):
            table.add_row(
                f.name,
                f"{f.confidence:.2f}",
                f.value or ""
            )
        
        console.print(table)
    
    # Text matching
    text_match = text_analyzer.match_against_query(
        property_analysis=text_analysis,
        query_text=query,
        required_features=requirements,
        feature_weight=0.7,
        semantic_weight=0.3
    )
    
    console.print(f"\n[bold]Text Matching Score:[/bold] {text_match.final_score:.3f}")
    console.print(f"  Feature match: {text_match.feature_match_score:.3f}")
    console.print(f"  Semantic match: {text_match.semantic_similarity_score:.3f}")
    
    # ============= VISION ANALYSIS =============
    console.print("\n[bold cyan]═══ VISION ANALYSIS (CLIP) ═══[/bold cyan]")
    
    images = property_data.get('images', [])
    if not images:
        console.print("[yellow]No images available[/yellow]")
        return
    
    vision_analyzer = VisionAnalyzer()
    
    # Select key images
    selected = vision_analyzer.select_key_images(images, [], max_images=3)
    console.print(f"\nSelected {len(selected)} images:")
    
    for idx, (url, purpose) in enumerate(selected):
        console.print(f"  {idx+1}. {purpose}: {url[-50:]}")
    
    # Save images if requested
    if save_images:
        console.print("\n[cyan]Saving images...[/cyan]")
        for idx, (url, purpose) in enumerate(selected):
            filepath = output_dir / f"image_{idx}_{purpose}.jpg"
            if save_image(url, filepath):
                console.print(f"  ✅ Saved: {filepath}")
    
    # Generate embeddings and compute similarity
    console.print("\n[cyan]Computing CLIP similarity...[/cyan]")
    
    vision_analysis = vision_analyzer.analyze_property_stage1(
        property_id=prop_id,
        image_urls=images,
        max_images=3
    )
    
    if vision_analysis.image_embeddings:
        console.print(f"Generated {len(vision_analysis.image_embeddings)} embeddings")
        
        # Compute similarity for each image
        table = Table(title="CLIP Similarity Scores")
        table.add_column("Image", style="cyan")
        table.add_column("Purpose", style="yellow")
        table.add_column("Similarity to Query", style="green")
        
        for idx, (url, purpose) in enumerate(selected[:len(vision_analysis.image_embeddings)]):
            embedding = vision_analysis.image_embeddings[idx]
            similarity = vision_analyzer.compute_image_text_similarity(
                embedding,
                query
            )
            table.add_row(
                f"Image {idx+1}",
                purpose,
                f"{similarity:.3f}"
            )
        
        console.print(table)
        
        # Average similarity
        avg_sim = vision_analysis.get_avg_embedding()
        if avg_sim is not None:
            overall_sim = vision_analyzer.compute_image_text_similarity(avg_sim, query)
            console.print(f"\n[bold]Overall CLIP Similarity:[/bold] {overall_sim:.3f}")
    
    # ============= COMBINED SCORING =============
    console.print("\n[bold cyan]═══ COMBINED SCORING ═══[/bold cyan]")
    
    # Compute combined score
    avg_vision_sim = 0.0
    if vision_analysis.get_avg_embedding() is not None:
        avg_vision_sim = vision_analyzer.compute_image_text_similarity(
            vision_analysis.get_avg_embedding(),
            query
        )
    
    combined_score = (
        text_match.feature_match_score * 0.5 +
        text_match.semantic_similarity_score * 0.3 +
        avg_vision_sim * 0.2
    )
    
    console.print(f"\n[bold]Score Breakdown:[/bold]")
    console.print(f"  Text features (50%):     {text_match.feature_match_score:.3f} → {text_match.feature_match_score * 0.5:.3f}")
    console.print(f"  Text semantic (30%):     {text_match.semantic_similarity_score:.3f} → {text_match.semantic_similarity_score * 0.3:.3f}")
    console.print(f"  Vision semantic (20%):   {avg_vision_sim:.3f} → {avg_vision_sim * 0.2:.3f}")
    console.print(f"  ─────────────────────────────")
    console.print(f"  [bold]Text-only score:[/bold]      {text_match.final_score:.3f}")
    console.print(f"  [bold]Text+Vision score:[/bold]    {combined_score:.3f}")
    console.print(f"  [bold]Difference:[/bold]           {(combined_score - text_match.final_score):.3f}")
    
    # ============= INTERPRETATION =============
    console.print("\n[bold cyan]═══ INTERPRETATION ═══[/bold cyan]")
    
    if combined_score > text_match.final_score:
        console.print("[green]✅ Vision improved the score![/green]")
        console.print("   CLIP found visual similarity to your query")
    elif combined_score < text_match.final_score:
        console.print("[yellow]⚠️  Vision reduced the score[/yellow]")
        console.print("   CLIP's generic similarity diluted specific text features")
        console.print("   This suggests: text features are more reliable than CLIP")
    else:
        console.print("[blue]➡️  Vision had no effect[/blue]")
    
    # Matched requirements
    console.print(f"\n[bold]Requirements Check:[/bold]")
    for req in requirements:
        matched = any(f.name == req.feature_name for f in text_match.matched_features)
        status = "✅" if matched else "❌"
        console.print(f"  {status} {req.feature_name} (importance: {req.importance})")
    
    if text_match.missing_requirements:
        console.print(f"\n[yellow]Missing: {', '.join(text_match.missing_requirements)}[/yellow]")
    
    console.print(f"\n[bold]Output directory:[/bold] {output_dir}")


def main():
    import sys
    
    # Get property ID from command line or use first available
    property_id = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Configuration
    query = "Local comercial Barcelona con entrada independiente, luz natural, máximo 300 mil euros"
    requirements = [
        QueryRequirement(feature_name="entrada_independiente", importance=1.0, required=True),
        QueryRequirement(feature_name="luz_natural", importance=0.8, required=False),
        QueryRequirement(feature_name="local_comercial", importance=0.9, required=False)
    ]
    
    console.print("[bold blue]🔍 Vision Analysis Debug Tool[/bold blue]\n")
    console.print(f"Query: {query}\n")
    
    # Load property
    console.print(f"Loading property {property_id or '(first available)'}...")
    prop = load_one_property(property_id)
    
    if not prop:
        console.print("[red]No property found![/red]")
        return
    
    # Analyze with full transparency
    analyze_property_debug(prop, query, requirements, save_images=True)


if __name__ == "__main__":
    main()