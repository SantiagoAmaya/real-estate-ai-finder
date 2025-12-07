"""
Debug detallado del análisis de visión
Muestra imágenes analizadas con descripciones basadas en query del usuario
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import requests
from PIL import Image
from io import BytesIO
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()

def download_and_save_image(url: str, save_path: Path) -> bool:
    """Download and save image locally"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        img.save(save_path)
        return True
    except Exception as e:
        console.print(f"[red]Failed to save {url}: {e}[/red]")
        return False

# Get query from command line or use default
import sys
if len(sys.argv) > 1:
    user_query = " ".join(sys.argv[1:])
else:
    user_query = "Local Barcelona entrada independiente, techos altos, espacio diáfano"

console.print(Panel.fit(
    f"[bold cyan]User Query:[/bold cyan]\n{user_query}",
    border_style="cyan"
))

# Load property
console.print("\n[cyan]Loading property data...[/cyan]")

data_dir = Path("data/raw")
property_data = None

for json_file in sorted(data_dir.glob("fotocasa_*.json")):
    try:
        with open(json_file) as f:
            data = json.load(f)
            props = data.get('properties', [])
            # Find one with many images
            for p in props:
                if len(p.get('images', [])) >= 5:
                    property_data = p
                    break
            if property_data:
                break
    except:
        continue

if not property_data:
    console.print("[red]No property found![/red]")
    sys.exit(1)

# Show property info
console.print(Panel.fit(
    f"[bold]Property ID:[/bold] {property_data['id']}\n"
    f"[bold]Title:[/bold] {property_data.get('title', 'N/A')[:80]}\n"
    f"[bold]Location:[/bold] {property_data.get('location', 'N/A')}\n"
    f"[bold]Price:[/bold] €{property_data.get('price', 0):,}\n"
    f"[bold]Total images:[/bold] {len(property_data['images'])}",
    border_style="blue"
))

# Create output directory
output_dir = Path(f"data/debug/{property_data['id']}")
output_dir.mkdir(parents=True, exist_ok=True)

# Select images
from src.property_analysis.vision_analyzer import VisionAnalyzer

analyzer = VisionAnalyzer(mode="claude_only")

all_images = property_data['images']
selected = analyzer.select_key_images(all_images, max_images=3)

console.print(f"\n[bold cyan]Images Selected (3 of {len(all_images)}):[/bold cyan]\n")

# Table of selected images
table = Table(show_header=True, header_style="bold magenta")
table.add_column("#", style="cyan", width=3)
table.add_column("Purpose", style="yellow", width=20)
table.add_column("URL", style="green")
table.add_column("Saved As", style="blue", width=25)

for idx, (url, purpose) in enumerate(selected, 1):
    filename = f"image_{idx}.jpg"
    filepath = output_dir / filename
    
    if download_and_save_image(url, filepath):
        saved = f"✅ {filename}"
    else:
        saved = "❌ Failed"
    
    url_short = url[-60:] if len(url) > 60 else url
    table.add_row(str(idx), f"Selected #{idx}", url_short, saved)

console.print(table)
console.print(f"\n[bold]Images saved to:[/bold] {output_dir}\n")

# Analyze with query-aware prompt
console.print("[cyan]Analyzing images with query-aware Claude Vision...[/cyan]\n")

# DYNAMIC PROMPT based on user query
dynamic_prompt = f"""Analiza esta foto de forma OBJETIVA y FACTUAL.

CONTEXTO - El usuario busca:
"{user_query}"

Describe lo que ves en la imagen que sea RELEVANTE para esta búsqueda,e.g.:

1. **Tipo de espacio**: Local comercial, piso, oficina, área específica
2. **Acceso**: Entrada desde calle (independiente) o portal compartido
3. **Techos**: Altura en metros (2.5m estándar, 3.0m medio-alto, 3.5m+ alto, 4m+ muy alto)
4. **Suelos**: Material exacto (cerámica, parquet, baldosa, hormigón)
5. **Luz natural**: Número de ventanas/puertas, tamaño
6. **Distribución**: Diáfano, compartimentado, espacios visibles
7. **Estado**: Nuevo/reformado/antiguo
8. **Otros elementos relevantes** para la búsqueda del usuario

Responde en maximo 2 párrafos FACTUALES, sin lenguaje comercial.
Máximo 120 palabras."""

# Generate descriptions
descriptions_parts = []

for idx, (url, purpose) in enumerate(selected, 1):
    try:
        console.print(f"  Analyzing image {idx}/3...")
        
        # Download image
        response = requests.get(url, timeout=10)
        img = Image.open(BytesIO(response.content))
        
        # Convert to base64
        import base64
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Call Claude with query-aware prompt
        from anthropic import Anthropic
        import os
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": dynamic_prompt
                    }
                ]
            }]
        )
        
        description = response.content[0].text.strip()
        descriptions_parts.append((idx, purpose, description))
        
    except Exception as e:
        console.print(f"[red]  Error: {e}[/red]")
        continue

# Display results
console.print("\n" + "="*80)
console.print("[bold green]QUERY-AWARE VISION ANALYSIS[/bold green]")
console.print("="*80 + "\n")

for idx, purpose, desc in descriptions_parts:
    console.print(Panel(
        desc,
        title=f"[bold cyan]Image {idx} - {purpose}[/bold cyan]",
        border_style="cyan"
    ))
    console.print()

# Statistics
console.print(f"[bold]Statistics:[/bold]")
console.print(f"  Query: {user_query}")
console.print(f"  Images analyzed: {len(descriptions_parts)}")
console.print(f"  Cost: €{len(descriptions_parts) * 0.025:.3f}")
console.print(f"\n[bold]View images:[/bold]")
console.print(f"  open {output_dir}")