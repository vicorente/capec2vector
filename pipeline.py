import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility,
)
import requests
import json
import os

# Configuración
XML_FILE = os.path.join("capec_latest", "capec_v3.9.xml")
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "capec_vectors"
MODEL_NAME = "all-MiniLM-L6-v2"  # Modelo de embeddings
OLLAMA_URL = "http://localhost:11434/api/generate"


def extract_attack_pattern(pattern, namespace):
    """Extrae la información relevante de un patrón de ataque"""
    try:
        # Obtener atributos básicos
        pattern_id = pattern.get("ID")
        pattern_name = pattern.get("Name")
        pattern_status = pattern.get("Status")
        pattern_abstraction = pattern.get("Abstraction")

        # Obtener elementos de texto
        description = pattern.find("capec:Description", namespace)
        description_text = (
            description.text if description is not None and description.text else ""
        )
        if description is not None and len(description) > 0:
            # Si hay elementos hijos, obtener el texto de todos los hijos
            description_text = "".join(description.itertext()).strip()

        # Obtener severidad y probabilidad
        likelihood = pattern.find("capec:Likelihood_Of_Attack", namespace)
        likelihood_text = likelihood.text if likelihood is not None else "Not Specified"

        severity = pattern.find("capec:Typical_Severity", namespace)
        severity_text = severity.text if severity is not None else "Not Specified"

        return {"id": pattern_id, "name": pattern_name, "description": description}

    except AttributeError as e:
        print(f"Error procesando patrón: {e}")
        return None


# Paso 1: Parsear el XML
def parse_capec_xml(xml_file):
    """Extrae patrones de ataque del archivo XML."""
    if not os.path.exists(xml_file):
        print(f"Error: El archivo XML no se encuentra en: {xml_file}")
        return

    # Parsear el XML
    tree = ET.parse(xml_file)
    root = tree.getroot()
    attack_patterns = []

    # Definir el namespace
    namespace = {"capec": root.tag.split("}")[0].strip("{")} if "}" in root.tag else ""
    print("Procesando archivo XML...")
    # Obtener el contenedor Attack_Patterns
    attack_patterns_container = root.find("capec:Attack_Patterns", namespace)

    # Iterar sobre todos los patrones de ataque dentro del contenedor
    for pattern in attack_patterns_container.findall("capec:Attack_Pattern", namespace):
        pattern_data = extract_attack_pattern(pattern, namespace)
        if pattern_data:
            attack_patterns.append(pattern_data)

    return attack_patterns


# Paso 2: Generar Embeddings
def generate_embeddings(attack_patterns, model_name):
    """Convierte texto en vectores."""
    model = SentenceTransformer(model_name)
    texts = [pattern["text"] for pattern in attack_patterns]
    embeddings = model.encode(texts, convert_to_tensor=False)
    return embeddings


# Paso 3: Configurar y Almacenar en Milvus
def setup_milvus_collection(collection_name, dim=384):
    """Configura la colección en Milvus."""
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

    if utility.has_collection(collection_name):
        Collection(collection_name).drop()

    # Definir esquema
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=5000),
    ]
    schema = CollectionSchema(fields=fields, description="CAPEC Attack Patterns")
    collection = Collection(collection_name, schema)

    # Crear índice
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128},
    }
    collection.create_index("embedding", index_params)
    collection.load()
    return collection


def store_in_milvus(attack_patterns, embeddings, collection):
    """Guarda vectores y metadatos en Milvus."""
    data = [
        [pattern["id"] for pattern in attack_patterns],
        [embedding.tolist() for embedding in embeddings],
        [pattern["name"] for pattern in attack_patterns],
        [pattern["description"] for pattern in attack_patterns],
    ]
    collection.insert(data)
    print(f"Almacenados {len(attack_patterns)} patrones en Milvus.")


# Paso 4: Buscar contexto en Milvus
def retrieve_context(query, collection, model, top_k=3):
    """Busca patrones relevantes en Milvus."""
    query_embedding = model.encode([query])[0].tolist()
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["id", "name", "description"],
    )

    context = []
    for hits in results:
        for hit in hits:
            pattern_info = (
                f"ID: {hit.entity.get('id')}, "
                f"Name: {hit.entity.get('name')}, "
                f"Description: {hit.entity.get('description')}"
            )
            context.append(pattern_info)

    return "\n\n".join(context)


# Paso 5: Generar respuesta con Ollama
def generate_response(query, context):
    """Envía la consulta y el contexto a Ollama."""
    prompt = (
        f"Basado en el siguiente contexto sobre ciberseguridad, responde la consulta:\n\n"
        f"Consulta: {query}\n\n"
        f"Contexto:\n{context}\n\n"
        f"Proporciona una explicación clara y detallada."
    )

    payload = {"model": "llama3", "prompt": prompt, "stream": False}

    response = requests.post(OLLAMA_URL, json=payload)
    if response.status_code == 200:
        return json.loads(response.text)["response"]
    else:
        return f"Error con Ollama: {response.status_code}"


# Pipeline completo
def capec_milvus_ollama_pipeline(xml_file, collection_name, model_name):
    """Integra Milvus y Ollama."""
    print("Iniciando pipeline...")

    # Parsear XML
    attack_patterns = parse_capec_xml(xml_file)
    print(f"Extraídos {len(attack_patterns)} patrones de ataque.")

    # Generar embeddings
    embeddings = generate_embeddings(attack_patterns, model_name)
    print(f"Generados {len(embeddings)} embeddings.")

    # Configurar y almacenar en Milvus
    collection = setup_milvus_collection(collection_name, dim=384)
    store_in_milvus(attack_patterns, embeddings, collection)

    # Ejemplo de consulta
    query = "Explícame cómo funciona un ataque de inundación en ciberseguridad."
    model = SentenceTransformer(model_name)
    context = retrieve_context(query, collection, model)
    print("Contexto recuperado:\n", context)

    response = generate_response(query, context)
    print("Respuesta generada:\n", response)

    return collection  # Retorna para consultas posteriores


# Ejecutar el pipeline
if __name__ == "__main__":
    if not os.path.exists(XML_FILE):
        print(f"Error: El archivo {XML_FILE} no existe.")
    else:
        collection = capec_milvus_ollama_pipeline(XML_FILE, COLLECTION_NAME, MODEL_NAME)
