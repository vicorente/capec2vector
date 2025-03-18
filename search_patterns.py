from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import argparse

# Configuración de Milvus
MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
COLLECTION_NAME = "capec_patterns"

def connect_to_milvus():
    """Establece conexión con Milvus"""
    try:
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        collection = Collection(COLLECTION_NAME)
        collection.load()
        return collection
    except Exception as e:
        print(f"Error al conectar con Milvus: {e}")
        raise

def search_patterns(query_text, top_k=5):
    """
    Realiza una búsqueda semántica de patrones de ataque basada en el texto de consulta.
    
    Args:
        query_text (str): Texto de búsqueda en lenguaje natural
        top_k (int): Número de resultados a retornar
    
    Returns:
        list: Lista de diccionarios con los patrones encontrados y sus puntuaciones
    """
    try:
        # Cargar el modelo de embeddings
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Generar embedding para la consulta
        query_embedding = model.encode(query_text)
        
        # Conectar a Milvus
        collection = connect_to_milvus()
        
        # Realizar búsqueda
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }
        
        results = collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["pattern_id", "name", "description"]
        )
        
        # Formatear resultados
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "pattern_id": hit.entity.get("pattern_id"),
                    "name": hit.entity.get("name"),
                    "description": hit.entity.get("description"),
                    "similarity_score": hit.score
                })
        
        return formatted_results
    
    except Exception as e:
        print(f"Error durante la búsqueda: {e}")
        raise
    finally:
        connections.disconnect("default")

def print_results(results):
    """Imprime los resultados de manera formateada"""
    print("\nResultados de la búsqueda:")
    print("=" * 80)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Patrón ID: {result['pattern_id']}")
        print(f"Nombre: {result['name']}")
        print(f"Puntuación de similitud: {1 - result['similarity_score']:.4f}")  # Convertimos distancia L2 a similitud
        print(f"Descripción: {result['description'][:200]}...")
        print("-" * 80)

def main():
    parser = argparse.ArgumentParser(description="Buscar patrones de ataque CAPEC usando similitud semántica")
    parser.add_argument("query", type=str, help="Texto de búsqueda en lenguaje natural")
    parser.add_argument("--top_k", type=int, default=5, help="Número de resultados a mostrar (default: 5)")
    
    args = parser.parse_args()
    
    try:
        results = search_patterns(args.query, args.top_k)
        print_results(results)
    except Exception as e:
        print(f"Error en la ejecución: {e}")

if __name__ == "__main__":
    main() 