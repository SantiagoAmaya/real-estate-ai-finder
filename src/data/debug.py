from curl_cffi import requests
from bs4 import BeautifulSoup
import re
import json

def diagnostico_profundo():
    url = "https://www.fotocasa.es/es/comprar/viviendas/madrid-capital/todas-las-zonas/l"
    
    print(f"🕵️ INICIANDO DIAGNÓSTICO FORENSE EN: {url}")
    print("-" * 60)

    try:
        session = requests.Session()
        # Usamos Chrome 120 para máxima compatibilidad
        response = session.get(
            url, 
            impersonate="chrome120",
            headers={
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                'Accept-Language': 'es-ES,es;q=0.9',
                'Referer': 'https://www.google.com/'
            },
            timeout=30
        )
    except Exception as e:
        print(f"❌ Error fatal de conexión: {e}")
        return

    print(f"📡 Estado HTTP: {response.status_code}")
    html = response.text
    soup = BeautifulSoup(html, 'html.parser')

    # 1. BUSQUEDA DE ESTRUCTURA HTML (Si hay cards visibles)
    # -----------------------------------------------------
    print("\n1️⃣  ANÁLISIS DE ESTRUCTURA HTML")
    articles = soup.find_all('article')
    cards_re = soup.find_all(class_=re.compile(r're-Card'))
    cards_sui = soup.find_all(class_=re.compile(r'sui-Card'))
    
    print(f"   - Etiquetas <article> encontradas: {len(articles)}")
    print(f"   - Clases 're-Card' encontradas: {len(cards_re)}")
    print(f"   - Clases 'sui-Card' encontradas: {len(cards_sui)}")

    if len(articles) == 0 and len(cards_re) == 0:
        print("   ⚠️  CONCLUSIÓN: La página no tiene anuncios renderizados en HTML. Depende de JavaScript/JSON.")
    else:
        print("   ✅ CONCLUSIÓN: Hay contenido HTML visible.")

    # 2. BÚSQUEDA DE VARIABLES DE DATOS (El Tesoro)
    # -----------------------------------------------------
    print("\n2️⃣  BÚSQUEDA DE VARIABLES JSON (HIDDEN DATA)")
    
    scripts = soup.find_all('script')
    targets = [
        "__INITIAL_PROPS__", 
        "__NEXT_DATA__", 
        "__INITIAL_STATE__", 
        "window.initialData",
        "utag_data"
    ]
    
    found_data = False
    
    for script in scripts:
        if not script.string: continue
        content = script.string
        
        for target in targets:
            if target in content:
                print(f"\n   🎯 ¡ENCONTRADO! Variable: '{target}'")
                found_data = True
                
                # Análisis de contexto: ¿Es JSON.parse o Objeto directo?
                start_idx = content.find(target)
                sample = content[start_idx:start_idx+150] # Primeros 150 caracteres
                print(f"      CONTEXTO: {sample} ...")
                
                if "JSON.parse" in sample:
                    print("      FORMATO: Stringified JSON (Requiere doble decodificación)")
                elif "=" in sample and "{" in sample:
                    print("      FORMATO: Objeto JS Directo")
                
                # Intentar ver si contiene la palabra clave 'realEstates' (inmuebles)
                if "realEstates" in content:
                    print("      ✅ CONTIENE DATOS DE INMUEBLES ('realEstates' detectado)")
                else:
                    print("      ⚠️  NO PARECE CONTENER INMUEBLES (Podría ser config)")

    if not found_data:
        print("\n   ❌ NO SE ENCONTRARON VARIABLES DE DATOS CONOCIDAS.")
        print("      Es posible que Fotocasa haya cambiado el nombre de la variable.")

    # 3. DETECCIÓN DE BLOQUEOS
    # -----------------------------------------------------
    print("\n3️⃣  ANÁLISIS DE BLOQUEO")
    page_title = soup.title.string.strip() if soup.title else "Sin título"
    print(f"   - Título de página: {page_title}")
    
    if "Robot" in page_title or "block" in html.lower() or "captcha" in html.lower():
        print("   🚨 ALERTA: Posible BLOQUEO detectado (Captcha/Botwall).")
    else:
        print("   ✅ No se detectan bloqueos obvios en el texto.")

    # 4. GUARDADO DE MUESTRA
    # -----------------------------------------------------
    with open("debug_forense.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("\n💾 HTML guardado en 'debug_forense.html' para inspección manual.")

if __name__ == "__main__":
    diagnostico_profundo()