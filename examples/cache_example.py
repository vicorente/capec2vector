#!/usr/bin/env python3
"""
Ejemplo de uso del sistema de caché para embeddings
"""
import sys
import time
from pathlib import Path
import numpy as np

# Añadir el directorio padre al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.cache import EmbeddingCache


def simulate_embedding_generation(text):
    """Simula la generación de un embedding (costosa operacionalmente)"""
    print(f"  🔄 Generando embedding para: '{text[:50]}...'")
    time.sleep(0.5)  # Simular tiempo de procesamiento
    # Crear un embedding simulado (en realidad sería de un modelo)
    return np.random.rand(768).astype(np.float32)


def example_basic_usage():
    """Ejemplo básico de uso del caché"""
    print("\n" + "="*80)
    print("Ejemplo 1: Uso básico del caché")
    print("="*80 + "\n")
    
    # Crear instancia de caché
    cache = EmbeddingCache(Path(".cache/examples"), ttl_seconds=3600)
    
    texts = [
        "SQL Injection is a code injection technique",
        "Cross-site scripting allows attackers to inject malicious scripts",
        "Buffer overflow occurs when data exceeds buffer boundaries",
    ]
    
    # Primera pasada: generar y cachear
    print("Primera pasada: generando embeddings...\n")
    for text in texts:
        start = time.time()
        
        # Intentar obtener del caché
        embedding = cache.get(text)
        
        if embedding is None:
            # No está en caché, generar
            embedding = simulate_embedding_generation(text)
            cache.set(text, embedding)
            print(f"  ✅ Cacheado ({time.time() - start:.2f}s)")
        else:
            print(f"  ⚡ Recuperado del caché ({time.time() - start:.4f}s)")
        print()
    
    # Segunda pasada: todo desde caché
    print("\nSegunda pasada: recuperando del caché...\n")
    for text in texts:
        start = time.time()
        embedding = cache.get(text)
        print(f"  ⚡ '{text[:50]}...' - {time.time() - start:.4f}s")
    
    print(f"\n✨ Reducción de tiempo: ~{0.5 * len(texts):.1f}s → ~0.001s")


def example_cache_stats():
    """Ejemplo de estadísticas del caché"""
    print("\n" + "="*80)
    print("Ejemplo 2: Estadísticas del caché")
    print("="*80 + "\n")
    
    cache = EmbeddingCache(Path(".cache/examples"), ttl_seconds=3600)
    
    # Obtener estadísticas
    stats = cache.get_stats()
    
    print("📊 Estadísticas del caché:")
    print(f"  Total de elementos: {stats['total_items']}")
    print(f"  Elementos activos: {stats['active_items']}")
    print(f"  Elementos expirados: {stats['expired_items']}")
    print(f"  Tamaño total: {stats['total_size_mb']} MB")
    print(f"  Directorio: {stats['cache_dir']}")


def example_cache_management():
    """Ejemplo de gestión del caché"""
    print("\n" + "="*80)
    print("Ejemplo 3: Gestión del caché")
    print("="*80 + "\n")
    
    cache = EmbeddingCache(Path(".cache/examples"), ttl_seconds=3600)
    
    # Añadir varios elementos
    print("Añadiendo elementos al caché...")
    for i in range(5):
        text = f"Pattern description {i}"
        embedding = np.random.rand(768).astype(np.float32)
        cache.set(text, embedding)
        print(f"  ✅ Item {i+1} cacheado")
    
    stats_before = cache.get_stats()
    print(f"\nAntes de limpiar: {stats_before['total_items']} elementos")
    
    # Limpiar caché
    print("\n🗑️  Limpiando caché...")
    cache.clear()
    
    stats_after = cache.get_stats()
    print(f"Después de limpiar: {stats_after['total_items']} elementos")


def example_ttl_expiration():
    """Ejemplo de expiración por TTL"""
    print("\n" + "="*80)
    print("Ejemplo 4: Expiración por TTL")
    print("="*80 + "\n")
    
    # Crear caché con TTL corto para demostración
    cache = EmbeddingCache(Path(".cache/examples_ttl"), ttl_seconds=2)
    
    text = "Test pattern with short TTL"
    embedding = np.random.rand(768).astype(np.float32)
    
    print("Guardando embedding con TTL de 2 segundos...")
    cache.set(text, embedding)
    
    # Inmediatamente después
    result = cache.get(text)
    print(f"  Inmediatamente: {'✅ Encontrado' if result is not None else '❌ No encontrado'}")
    
    # Esperar 1 segundo
    print("\nEsperando 1 segundo...")
    time.sleep(1)
    result = cache.get(text)
    print(f"  Después de 1s: {'✅ Encontrado' if result is not None else '❌ No encontrado'}")
    
    # Esperar otro segundo (total 2s, debería expirar)
    print("\nEsperando otro segundo más...")
    time.sleep(1.5)
    result = cache.get(text)
    print(f"  Después de 2.5s: {'✅ Encontrado' if result is not None else '❌ Expirado'}")


def main():
    """Ejecuta todos los ejemplos"""
    print("\n🎯 Ejemplos de uso del sistema de caché\n")
    
    try:
        example_basic_usage()
        input("\nPresiona Enter para continuar...")
        
        example_cache_stats()
        input("\nPresiona Enter para continuar...")
        
        example_cache_management()
        input("\nPresiona Enter para continuar...")
        
        example_ttl_expiration()
        
        print("\n" + "="*80)
        print("✅ Todos los ejemplos completados exitosamente!")
        print("="*80)
        
        print("\n💡 Tips:")
        print("  - Usa el caché para embeddings costosos de generar")
        print("  - Ajusta el TTL según tus necesidades (default: 3600s)")
        print("  - Monitorea el tamaño del caché con get_stats()")
        print("  - Limpia el caché periódicamente si crece mucho")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
