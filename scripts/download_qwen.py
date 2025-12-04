"""
Descarga Qwen2-VL con retry automático
"""
from huggingface_hub import snapshot_download
import time
import sys

def download_with_retry(max_retries=10):
    """Descarga con reintentos automáticos"""
    
    model_name = "Qwen/Qwen2-VL-7B-Instruct"
    cache_dir = "data/cache/qwen"
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"\n{'='*60}")
            print(f"Intento {attempt}/{max_retries}")
            print(f"{'='*60}\n")
            
            # snapshot_download tiene resume automático
            snapshot_download(
                repo_id=model_name,
                cache_dir=cache_dir,
                resume_download=True,  # Resume si falla
                max_workers=4,  # Descargas paralelas
                local_files_only=False
            )
            
            print("\n✅ Descarga completa!")
            return True
            
        except Exception as e:
            print(f"\n⚠️  Error en intento {attempt}: {e}")
            
            if attempt < max_retries:
                wait_time = min(60, 10 * attempt)  # Backoff exponencial
                print(f"Reintentando en {wait_time} segundos...")
                time.sleep(wait_time)
            else:
                print(f"\n❌ Falló después de {max_retries} intentos")
                return False
    
    return False

if __name__ == "__main__":
    print("Descargando Qwen2-VL-7B-Instruct...")
    print("Puedes cancelar (Ctrl+C) y reiniciar sin perder progreso\n")
    
    success = download_with_retry(max_retries=10)
    sys.exit(0 if success else 1)