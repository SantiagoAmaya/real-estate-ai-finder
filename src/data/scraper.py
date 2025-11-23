"""
Fotocasa Scraper - V9 FINAL (Estructura Real Confirmada)
Basado en el análisis real del JSON
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
    
    def save(self, filepath: Optional[Path] = None) -> Path:
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = Path(f"data/raw/fotocasa_{timestamp}.json")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "properties": [p.to_dict() for p in self.properties],
                "metadata": {
                    "total_found": self.total_found,
                    "time": self.scraping_time_seconds
                }
            }, f, indent=2, ensure_ascii=False)
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
        type_slug = "viviendas" if property_type in ["piso", "casa", "vivienda"] else property_type
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
            json_string_real = json.loads(f'"{json_string_escaped}"')
            data = json.loads(json_string_real)
            
            # 4. Navegar el JSON
            real_estates = data.get('initialSearch', {}).get('result', {}).get('realEstates', [])
            
            if not real_estates:
                logger.error("No 'realEstates' found in JSON")
                return []

            logger.info(f"Found {len(real_estates)} items in realEstates")
            
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
        Estructura confirmada del item:
        - 'id': int/str
        - 'detail': dict con clave 'es-ES' (no 'es')
        - 'features': dict con 'surface', 'rooms', 'bathrooms'
        - 'price': dict con 'amount'
        - 'address': dict con 'municipality', 'district', etc.
        - 'location': string (no dict)
        - 'multimedia': lista de dicts con 'type' y 'url'
        - 'phone': string
        - 'description': string
        """
        try:
            if not isinstance(item, dict):
                logger.warning(f"Item is not a dict, type: {type(item)}")
                return None
            
            # ID
            pid = str(item.get('id', ''))
            if not pid:
                return None
            
            # URL - La clave es 'detail' con subclave 'es-ES'
            detail_dict = item.get('detail', {})
            detail_url = None
            if isinstance(detail_dict, dict):
                # Intentar ambas variantes
                detail_url = detail_dict.get('es-ES') or detail_dict.get('es')
            
            full_url = self.BASE_URL + detail_url if detail_url else ""
            
            # Características - 'features' es un dict
            features = item.get('features', {})
            size = 0
            rooms = 0
            bathrooms = 0
            
            if isinstance(features, dict):
                size = float(features.get('surface', 0) or 0)
                rooms = int(features.get('rooms', 0) or 0)
                bathrooms = int(features.get('bathrooms', 0) or 0)
            
            # Precio - 'price' es un dict con 'amount'
            price_data = item.get('price', {})
            price = 0
            if isinstance(price_data, dict):
                price = int(price_data.get('amount', 0) or 0)
            
            # Imágenes - 'multimedia' es una lista
            images = []
            multimedia_list = item.get('multimedia', [])
            if isinstance(multimedia_list, list):
                for media in multimedia_list:
                    if isinstance(media, dict):
                        media_type = str(media.get('type', '')).upper()
                        if media_type == 'IMAGE':
                            img_url = media.get('url') or media.get('src')
                            if img_url:
                                images.append(img_url)
            
            # Ubicación - 'address' es un dict
            address_data = item.get('address', {})
            location_parts = []
            
            if isinstance(address_data, dict):
                municipality = address_data.get('municipality', '')
                district = address_data.get('district', '')
                neighborhood = address_data.get('neighborhood', '')
                
                if neighborhood:
                    location_parts.append(neighborhood)
                if district and district != neighborhood:
                    location_parts.append(district)
                if municipality:
                    location_parts.append(municipality)
            
            location_str = ', '.join(filter(None, location_parts)) or 'Madrid'
            
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
    logger.info("🚀 Starting Fotocasa Scraper V9 FINAL...")
    
    result = scraper.scrape_properties(
        location="barcelona-capital", 
        property_type="vivienda",
        operation="comprar",
        max_pages=2
    )
    
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
        
        filepath = result.save()
        print(f"\n💾 Saved to: {filepath}")
        
        # Descomentar para descargar imágenes
        # print("\n📥 Downloading images...")
        # scraper.download_images(result.properties)
    else:
        print("\n⚠️  No properties found!")