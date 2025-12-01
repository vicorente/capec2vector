#!/usr/bin/env python3
"""
Ejemplo de búsqueda de patrones CAPEC usando la API REST
"""
import requests
import json
import sys


def search_patterns(query, top_k=5, api_url="http://localhost:8000"):
    """
    Busca patrones CAPEC usando búsqueda semántica.
    
    Args:
        query: Consulta en lenguaje natural
        top_k: Número de resultados a retornar
        api_url: URL base de la API
    """
    print(f"\n{'='*80}")
    print(f"Buscando: '{query}'")
    print(f"{'='*80}\n")
    
    try:
        response = requests.post(
            f"{api_url}/search",
            json={"query": query, "top_k": top_k},
            timeout=30
        )
        
        if response.status_code == 200:
            results = response.json()
            
            print(f"Encontrados {len(results.get('results', []))} patrones:\n")
            
            for i, pattern in enumerate(results.get("results", []), 1):
                print(f"{i}. {pattern['name']}")
                print(f"   ID: {pattern['pattern_id']}")
                print(f"   Similitud: {pattern.get('similarity_score', 0):.3f}")
                print(f"   Severidad: {pattern.get('typical_severity', 'N/A')}")
                print(f"   Probabilidad: {pattern.get('likelihood_of_attack', 'N/A')}")
                
                # Mostrar descripción truncada
                desc = pattern.get('description', '')
                if len(desc) > 150:
                    desc = desc[:150] + "..."
                print(f"   Descripción: {desc}")
                print()
        else:
            print(f"Error {response.status_code}: {response.text}")
            return None
            
        return results
        
    except requests.exceptions.ConnectionError:
        print("Error: No se puede conectar a la API.")
        print("Asegúrate de que el servidor está corriendo:")
        print("  uvicorn ollama_milvus_bridge:app --reload")
        sys.exit(1)
    except Exception as e:
        print(f"Error inesperado: {e}")
        sys.exit(1)


def main():
    """Función principal con ejemplos de búsqueda"""
    
    # Ejemplos de búsquedas
    examples = [
        "SQL injection attacks on web applications",
        "buffer overflow vulnerabilities",
        "cross-site scripting XSS",
        "privilege escalation techniques",
        "denial of service attacks"
    ]
    
    if len(sys.argv) > 1:
        # Usar la consulta de línea de comandos
        query = " ".join(sys.argv[1:])
        search_patterns(query, top_k=5)
    else:
        # Ejecutar ejemplos
        print("\n🔍 Ejecutando ejemplos de búsqueda...\n")
        
        for query in examples[:2]:  # Solo los primeros 2 para el ejemplo
            search_patterns(query, top_k=3)
            input("Presiona Enter para continuar...")
        
        print("\n💡 Tip: Puedes ejecutar búsquedas personalizadas:")
        print("  python search_example.py 'tu consulta aquí'")


if __name__ == "__main__":
    main()
