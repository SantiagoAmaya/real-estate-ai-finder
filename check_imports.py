import ast
import os
import sys

# Configuración
ROOT_DIR = os.getcwd()
ENTRY_POINT = "backend/main.py"  # Tu punto de entrada

visited_files = set()
imports_found = set()

def resolve_import_path(module_name, current_file_path):
    """Convierte 'src.data.scraper' en 'src/data/scraper.py'"""
    parts = module_name.split('.')
    
    # Caso base: ruta relativa a la raiz
    potential_path = os.path.join(ROOT_DIR, *parts) + ".py"
    if os.path.exists(potential_path):
        return potential_path
    
    # Caso directorio (ej: src/data/__init__.py)
    potential_init = os.path.join(ROOT_DIR, *parts, "__init__.py")
    if os.path.exists(potential_init):
        return potential_init
        
    return None

def scan_file(file_path):
    if file_path in visited_files:
        return
    visited_files.add(file_path)
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except Exception as e:
        print(f"Error leyendo {file_path}: {e}")
        return

    # Buscar imports
    for node in ast.walk(tree):
        module = None
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module = node.module
                # Manejar imports relativos (from . import x) es complejo,
                # asumimos estructura absoluta 'src.' para simplificar
        
        if module and (module.startswith('src') or module.startswith('backend')):
            resolved_path = resolve_import_path(module, file_path)
            if resolved_path:
                imports_found.add(resolved_path)
                scan_file(resolved_path)

print(f"🔍 Rastreando dependencias desde: {ENTRY_POINT}...\n")
scan_file(os.path.join(ROOT_DIR, ENTRY_POINT))

print("✅ ARCHIVOS STRICTAMENTE NECESARIOS:")
# Ordenar y limpiar rutas para mostrar
sorted_files = sorted([os.path.relpath(p, ROOT_DIR) for p in visited_files])
for f in sorted_files:
    print(f"  - {f}")

print("\n❌ ARCHIVOS EN SRC QUE NO SE USAN (CANDIDATOS A IGNORAR):")
all_src_files = []
for root, dirs, files in os.walk(os.path.join(ROOT_DIR, "src")):
    for file in files:
        if file.endswith(".py"):
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, ROOT_DIR)
            if full_path not in visited_files:
                print(f"  - {rel_path}")