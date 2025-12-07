"""
Fotocasa Scraper - V10 MLOps Enhanced
Versión mejorada con metadata completa para tracking y reproducibilidad
"""
import json
import time
import random
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass, asdict, field

from curl_cffi import requests
from bs4 import BeautifulSoup
from loguru import logger

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Property:
    id: str
    url: str
    title: str
    price: int
    location: str
    size_m2: Optional[float] = None
    rooms: Optional[int] = None
    bathrooms: Optional[int] = None
    description: str = ""
    phone: str = ""
    images: List[str] = field(default_factory=list)
    scraped_at: str = field(default_factory=lambda: datetime.now().isoformat())
    source: str = "fotocasa"
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    

@dataclass
class ScraperResult:
    properties: List[Property]
    total_found: int
    scraping_time_seconds: float

    @staticmethod
    def _clean_text(text: str) -> str:
        """Limpia texto de caracteres problemáticos"""
        if not text:
            return text
        
        # Eliminar surrogates y caracteres de control
        cleaned = text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
        
        # Eliminar caracteres de control excepto newline/tab
        cleaned = ''.join(char for char in cleaned if ord(char) >= 32 or char in '\n\t')
        
        return cleaned

    def _clean_property(self, prop: Property) -> dict:
        """Limpia una property antes de serializar"""
        prop_dict = prop.to_dict()
        
        # Campos de texto a limpiar
        text_fields = ['title', 'description', 'location', 'phone']
        
        for field in text_fields:
            if field in prop_dict and isinstance(prop_dict[field], str):
                prop_dict[field] = self._clean_text(prop_dict[field])
        
        return prop_dict
    
    def save(
        self, 
        filepath: Optional[Path] = None,
        scraper_version: str = "v0.1",
        filters_used: Optional[Dict] = None
    ) -> Path:
        """
        Guarda resultados con metadata completa para tracking MLOps
        
        Args:
            filepath: Path personalizado (opcional)
            scraper_version: Versión del scraper para tracking de cambios
            filters_used: Diccionario con filtros aplicados en el scraping
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = Path(f"data/raw/fotocasa_{timestamp}.json")
        else:
            # AÑADIR: Si es directorio, generar filename
            filepath = Path(filepath)
            if filepath.is_dir() or not filepath.suffix:  # Es directorio o sin extensión
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = filepath / f"fotocasa_{timestamp}.json"
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Metadata completa para MLOps
        metadata = {
            "scraping": {
                "timestamp": datetime.now().isoformat(),
                "total_found": self.total_found,
                "scraping_time_seconds": self.scraping_time_seconds,
                "scraper_version": scraper_version
            },
            "filters": filters_used or {},
            "data_quality": {
                "properties_with_images": sum(1 for p in self.properties if p.images),
                "properties_with_description": sum(1 for p in self.properties if p.description),
                "properties_with_phone": sum(1 for p in self.properties if p.phone),
                "avg_images_per_property": round(
                    sum(len(p.images) for p in self.properties) / len(self.properties), 2
                ) if self.properties else 0,
                "properties_with_complete_location": sum(
                    1 for p in self.properties 
                    if p.location and p.location != "Unknown"
                )
            },
            "statistics": {
                "avg_price": round(
                    sum(p.price for p in self.properties) / len(self.properties), 2
                ) if self.properties else 0,
                "min_price": min((p.price for p in self.properties), default=0),
                "max_price": max((p.price for p in self.properties), default=0),
                "avg_size_m2": round(
                    sum(p.size_m2 for p in self.properties if p.size_m2) / 
                    sum(1 for p in self.properties if p.size_m2), 2
                ) if any(p.size_m2 for p in self.properties) else 0,
            }
        }

        print("Checking for problematic properties...")
        for i, p in enumerate(self.properties):
            try:
                json.dumps(p.to_dict(), ensure_ascii=False)
            except UnicodeEncodeError as e:
                print(f"⚠️  Property {i} ({p.id}) has encoding issues:")
                print(f"   Title: {p.title[:50]}...")
                print(f"   Error: {e}")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "properties": [self._clean_property(p) for p in self.properties],
                "metadata": metadata
            }, f, indent=2, ensure_ascii=False)
        
        logger.success(f"Saved {len(self.properties)} properties to {filepath}")
        logger.info(f"Data quality: {metadata['data_quality']}")
        
        return filepath

# ============================================================================
# FOTOCASA SCRAPER ENGINE
# ============================================================================

class FotocasaScraper:
    
    BASE_URL = "https://www.fotocasa.es"
    
    def __init__(self):
        self.session = requests.Session()

    def _get_headers(self) -> Dict[str, str]:
        return {
            'authority': 'www.fotocasa.es',
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'accept-language': 'es-ES,es;q=0.9',
            'cache-control': 'no-cache',
            'pragma': 'no-cache',
            'sec-ch-ua': '"Chromium";v="120", "Google Chrome";v="120", "Not-A.Brand";v="99"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
            'sec-fetch-dest': 'document',
            'sec-fetch-mode': 'navigate',
            'sec-fetch-site': 'none',
            'sec-fetch-user': '?1',
            'upgrade-insecure-requests': '1',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }

    
    def scrape_properties(
        self,
        location: str = "madrid-capital",
        property_type: str = "vivienda", 
        operation: str = "comprar",
        max_pages: int = 1,
        min_price: Optional[int] = None,
        max_price: Optional[int] = None,
        min_size_m2: Optional[int] = None,
        min_rooms: Optional[int] = None,  
        max_results: Optional[int] = None,
        ) -> ScraperResult:
        
        start_time = time.time()
        properties = []
        
        # 1. WARMUP
        try:
            logger.info("Warming up session...")
            self.session.get(self.BASE_URL, headers=self._get_headers(), impersonate="chrome120")
            time.sleep(1.5)
        except Exception as e:
            logger.error(f"Warmup error: {e}")

        # 2. LOOP
        for page_num in range(1, max_pages + 1):
            url = self._build_url(location, property_type, operation, page_num)
            logger.info(f"Scraping page {page_num}: {url}")
            
            try:
                response = self.session.get(
                    url, 
                    headers=self._get_headers(), 
                    impersonate="chrome120",
                    timeout=30
                )
                
                if response.status_code != 200:
                    logger.error(f"Error {response.status_code} on page {page_num}")
                    break
                
                # Parse properties
                page_props = self._extract_from_initial_props_manual(response.text)
                
                if not page_props:
                    logger.warning(f"No properties found on page {page_num}.")
                    break
                
                # Filtrar propiedades
                if any([min_price, max_price, min_size_m2, min_rooms]):
                    page_props = self._filter_properties(
                        page_props, 
                        min_price, 
                        max_price, 
                        min_size_m2, 
                        min_rooms
                    )

                properties.extend(page_props)
                logger.success(f"Found {len(page_props)} items on page {page_num}. Total: {len(properties)}")
                
                time.sleep(random.uniform(2.0, 4.0))
                
            except Exception as e:
                logger.error(f"Error scraping page {page_num}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                break

        return ScraperResult(
            properties=properties,
            total_found=len(properties),
            scraping_time_seconds=round(time.time() - start_time, 2)
        )

    def _build_url(self, location, property_type, operation, page) -> str:
        if property_type in ["piso", "casa", "vivienda"]:
            type_slug = "viviendas"
        elif property_type == "local":
            type_slug = "locales"  # ← FIX: plural
        else:
            type_slug = property_type
        
        base = f"/es/{operation}/{type_slug}/{location}/todas-las-zonas/l"
        if page > 1:
            base += f"/{page}"
        return f"{self.BASE_URL}{base}"

    def _extract_from_initial_props_manual(self, html: str) -> List[Property]:
        """Extrae el JSON usando parser manual"""
        soup = BeautifulSoup(html, 'html.parser')
        properties = []
        
        # 1. Localizar el script
        script_tag = soup.find('script', id='sui-scripts')
        if not script_tag or not script_tag.string:
            logger.error("Script tag 'sui-scripts' not found")
            return []

        content = script_tag.string
        marker = 'window.__INITIAL_PROPS__ = JSON.parse("'
        
        start_idx = content.find(marker)
        if start_idx == -1:
            logger.error("Marker not found in script")
            return []
            
        # Mover al inicio del String JSON
        current_idx = start_idx + len(marker)
        
        # 2. Parser Manual (State Machine)
        escaped = False
        json_string_end_idx = -1
        
        for i in range(current_idx, len(content)):
            char = content[i]
            
            if escaped:
                escaped = False
                continue
                
            if char == '\\':
                escaped = True
                continue
                
            if char == '"':
                json_string_end_idx = i
                break
        
        if json_string_end_idx == -1:
            logger.error("Could not find end of JSON string")
            return []

        json_string_escaped = content[current_idx:json_string_end_idx]
        
        try:
            # 3. Desescapar y Parsear
            json_string = json_string_escaped.encode('utf-8').decode('unicode_escape')
            data = json.loads(json_string)
            
            # 4. Navegar hasta 'realEstates'
            real_estates = (
                data.get('initialSearch', {})
                    .get('result', {})
                    .get('realEstates', [])
            )
            
            if not real_estates:
                logger.warning("'realEstates' is empty or not found")
                return []
            
            # 5. Mapear propiedades
            for idx, item in enumerate(real_estates):
                try:
                    prop = self._map_json_to_property(item)
                    if prop:
                        properties.append(prop)
                except Exception as e:
                    logger.error(f"Error mapping property {idx}: {e}")
                    # Continuar con el siguiente
                    
        except Exception as e:
            logger.error(f"Error parsing JSON: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
        return properties

    def _map_json_to_property(self, item) -> Optional[Property]:
        """
        Mapea basándose en la estructura REAL del JSON de Fotocasa
        """
        try:
            if not isinstance(item, dict):
                logger.warning(f"Item is not a dict, type: {type(item)}")
                return None
            
            # ID
            pid = str(item.get('id', ''))
            if not pid:
                return None
            
            # URL
            detail_dict = item.get('detail', {})
            detail_url = None
            if isinstance(detail_dict, dict):
                detail_url = detail_dict.get('es-ES') or detail_dict.get('es')
            
            full_url = self.BASE_URL + detail_url if detail_url else ""
            
            # PRECIO - Usar 'rawPrice' directamente (es un número)
            price = int(item.get('rawPrice', 0) or 0)
            
            # FEATURES - Es una LISTA, no un dict
            features_list = item.get('features', [])
            size = 0.0
            rooms = 0
            bathrooms = 0
            
            if isinstance(features_list, list):
                for feature in features_list:
                    if not isinstance(feature, dict):
                        continue
                    
                    key = feature.get('key', '')
                    value = feature.get('value', 0)
                    
                    if key == 'surface':
                        size = float(value or 0)
                    elif key == 'rooms':
                        rooms = int(value or 0)
                    elif key == 'bathrooms':
                        bathrooms = int(value or 0)
            
            # Imágenes
            images = []
            multimedia_list = item.get('multimedia', [])
            if isinstance(multimedia_list, list):
                for media in multimedia_list:
                    if isinstance(media, dict):
                        media_type = str(media.get('type', '')).lower()
                        if media_type == 'image':
                            img_url = media.get('src') or media.get('url')
                            if img_url:
                                images.append(img_url)
            
            # Ubicación
            address_data = item.get('address', {})
            location_parts = []
            
            if isinstance(address_data, dict):
                municipality = (address_data.get('municipality') or '').strip()
                district = (address_data.get('district') or '').strip()
                neighborhood = (address_data.get('neighborhood') or '').strip()
                
                if neighborhood:
                    location_parts.append(neighborhood)
                if district and district != neighborhood:
                    location_parts.append(district)
                if municipality:
                    location_parts.append(municipality)
            
            location_str = ', '.join(filter(None, location_parts)) or 'Unknown'
            
            # Descripción y teléfono
            description = str(item.get('description', ''))
            phone = str(item.get('phone', ''))
            
            # Título
            title = description[:100] if description else f"Inmueble en {location_str}"

            return Property(
                id=pid,
                url=full_url,
                title=title,
                price=price,
                location=location_str,
                size_m2=size if size > 0 else None,
                rooms=rooms if rooms > 0 else None,
                bathrooms=bathrooms if bathrooms > 0 else None,
                description=description,
                phone=phone,
                images=images
            )
            
        except Exception as e:
            logger.error(f"Error in _map_json_to_property: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
        
    def _filter_properties(
        self,
        properties: List[Property],
        min_price: Optional[int],
        max_price: Optional[int],
        min_size: Optional[int],
        min_rooms: Optional[int],
        ) -> List[Property]:
        """Filtrar propiedades (solo aplica filtro si el dato existe)"""
        filtered = []
        for p in properties:
            # Precio (siempre existe)
            if min_price and p.price < min_price:
                continue
            if max_price and p.price > max_price:
                continue
            
            # Tamaño (solo filtrar si existe)
            if min_size and p.size_m2 and p.size_m2 < min_size:
                continue
            
            # Habitaciones (solo filtrar si existe)
            if min_rooms and p.rooms and p.rooms < min_rooms:
                continue
            
            filtered.append(p)
        
        return filtered    
    
    def download_images(self, properties: List[Property], base_dir: str = "data/images"):
        base_path = Path(base_dir)
        base_path.mkdir(parents=True, exist_ok=True)
        
        if not properties:
            return

        logger.info(f"Downloading images for {len(properties)} properties...")
        
        count = 0
        for prop in properties:
            if not prop.images:
                continue
            
            safe_id = "".join([c for c in prop.id if c.isalnum() or c in ('-','_')])
            prop_dir = base_path / safe_id
            prop_dir.mkdir(exist_ok=True)
            
            for idx, img_url in enumerate(prop.images[:3]):
                try:
                    resp = self.session.get(img_url, impersonate="chrome120", timeout=10)
                    if resp.status_code == 200:
                        with open(prop_dir / f"{idx}.jpg", 'wb') as f:
                            f.write(resp.content)
                except Exception:
                    pass
            count += 1
            time.sleep(0.1)
            
        logger.success(f"Images downloaded for {count} properties.")


if __name__ == "__main__":
    scraper = FotocasaScraper()
    logger.info("🚀 Starting Fotocasa Scraper V10 MLOps Enhanced...")
    
    # Definir filtros para tracking
    filters = {
        "location": "barcelona-capital",
        "property_type": "vivienda",
        "operation": "comprar",
        "min_rooms": 2,
        "max_price": 500000,
        "min_size_m2": 40,
        "max_pages": 1
    }
    
    result = scraper.scrape_properties(**filters)
    
    print("\n" + "=" * 80)
    print(f"✅ RESULTS: Found {len(result.properties)} properties in {result.scraping_time_seconds}s")
    print("=" * 80)
    
    if len(result.properties) > 0:
        print("\n📋 Sample properties:")
        for i, p in enumerate(result.properties[:5]):
            print(f"\n  {i+1}. {p.price:,}€ | {p.size_m2}m² | {p.rooms} hab | {p.bathrooms} baños")
            print(f"     📍 {p.location}")
            print(f"     🖼️  {len(p.images)} images | 📞 {p.phone or 'N/A'}")
            print(f"     🔗 {p.url[:80]}...")
        
        filepath = result.save(
            scraper_version="v0.1", 
            filters_used=filters
        )
        print(f"\n💾 Saved to: {filepath}")
    else:
        print("\n⚠️  No properties found!")