"""
Data Quality Validator para Fotocasa
"""
import json
from pathlib import Path
import great_expectations as gx

class FotocasaValidator:
    def __init__(self):
        self.context = gx.get_context()
    
    def validate_json(self, json_path: Path) -> dict:
        """Valida un archivo JSON de scraping"""
        
        # Cargar datos
        with open(json_path) as f:
            data = json.load(f)
        
        properties = data.get('properties', [])
        metadata = data.get('metadata', {})
        
        # Validaciones
        results = {
            'file': json_path.name,
            'total_properties': len(properties),
            'validations': {}
        }
        
        # 1. Mínimo de propiedades
        results['validations']['min_properties'] = {
            'passed': len(properties) >= 3,
            'value': len(properties),
            'expected': '>= 3'
        }
        
        # 2. Campos requeridos presentes
        required_fields = ['id', 'price', 'location', 'url']
        for field in required_fields:
            missing = sum(1 for p in properties if not p.get(field))
            results['validations'][f'has_{field}'] = {
                'passed': missing == 0,
                'value': f"{len(properties) - missing}/{len(properties)}",
                'expected': 'all'
            }
        
        # 3. Calidad de imágenes
        with_images = sum(1 for p in properties if p.get('images'))
        results['validations']['has_images'] = {
            'passed': with_images / len(properties) >= 0.90,
            'value': f"{with_images/len(properties):.1%}",
            'expected': '>= 90%'
        }
        
        # 4. Calidad de descripciones
        with_desc = sum(1 for p in properties if p.get('description'))
        results['validations']['has_description'] = {
            'passed': with_desc / len(properties) >= 0.85,
            'value': f"{with_desc/len(properties):.1%}",
            'expected': '>= 85%'
        }
        
        # 5. Precios válidos
        valid_prices = sum(1 for p in properties 
                          if 50000 <= p.get('price', 0) <= 5000000)
        results['validations']['valid_prices'] = {
            'passed': valid_prices == len(properties),
            'value': f"{valid_prices}/{len(properties)}",
            'expected': 'all in range 50k-5M€'
        }
        
        # Summary
        results['passed'] = all(v['passed'] for v in results['validations'].values())
        results['passed_count'] = sum(v['passed'] for v in results['validations'].values())
        results['total_checks'] = len(results['validations'])
        
        return results
    
    def print_report(self, results: dict):
        """Imprime reporte legible"""
        print(f"\n{'='*60}")
        print(f"📊 Data Quality Report: {results['file']}")
        print(f"{'='*60}")
        print(f"Total properties: {results['total_properties']}")
        print(f"Checks passed: {results['passed_count']}/{results['total_checks']}")
        print(f"\nStatus: {'✅ PASSED' if results['passed'] else '❌ FAILED'}\n")
        
        for check, data in results['validations'].items():
            icon = "✅" if data['passed'] else "❌"
            print(f"{icon} {check:.<40} {data['value']} (expected: {data['expected']})")
        
        print(f"{'='*60}\n")
        return results['passed']