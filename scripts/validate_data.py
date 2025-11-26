#!/usr/bin/env python3
"""
Valida calidad de datos scrapeados
Uso: python scripts/validate_data.py [archivo.json]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_quality.validator import FotocasaValidator

def main():
    validator = FotocasaValidator()
    
    # Si se pasa archivo específico
    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])
        results = validator.validate_json(json_path)
        passed = validator.print_report(results)
        sys.exit(0 if passed else 1)
    
    # Validar todos los archivos en data/raw/
    data_dir = Path("data/raw")
    json_files = sorted(data_dir.glob("fotocasa_*.json"))
    
    print(f"🔍 Validating {len(json_files)} files...\n")
    
    all_results = []
    for json_file in json_files:
        results = validator.validate_json(json_file)
        all_results.append(results)
        
        # Solo imprimir si falla
        if not results['passed']:
            validator.print_report(results)
    
    # Summary
    passed = sum(r['passed'] for r in all_results)
    print(f"\n{'='*60}")
    print(f"📊 SUMMARY: {passed}/{len(all_results)} files passed validation")
    print(f"{'='*60}\n")
    
    sys.exit(0 if passed == len(all_results) else 1)

if __name__ == "__main__":
    main()