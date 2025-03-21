import xml.etree.ElementTree as ET
import os
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import numpy as np
import logging
from datetime import datetime

# Configura el logging básico con:
# - Nivel INFO para los mensajes
# - Formato que incluye fecha/hora, nivel y mensaje
# - Formato de fecha personalizado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# Crea un logger específico para este módulo
logger = logging.getLogger(__name__)

# docker run -d --name milvus_standalone -p 19530:19530 -p 19121:19121 milvusdb/milvus:latest

# Definir la ruta del archivo como una constante
XML_FILE_PATH = os.path.join("capec_latest", "capec_v3.9.xml")

# Configuración de Milvus
MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
COLLECTION_NAME = "capec_patterns"
DIMENSION = 384  # Dimensión del embedding de all-MiniLM-L6-v2

def create_milvus_collection():
    """Crea la colección en Milvus"""
    try:
        # Conectar a Milvus
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        logger.info("Conectado a Milvus")
        
        # Eliminar colección si existe
        if utility.has_collection(COLLECTION_NAME):
            logger.info(f"Eliminando colección existente {COLLECTION_NAME}...")
            utility.drop_collection(COLLECTION_NAME)
        
        # Definir el esquema de la colección
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="pattern_id", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="status", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="abstraction", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="typical_severity", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="typical_likelihood", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="typical_attack_vectors", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="typical_attack_prerequisites", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="typical_resources_required", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="typical_attack_mitigations", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="typical_attack_examples", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)
        ]
        
        schema = CollectionSchema(fields=fields, description="CAPEC Attack Patterns")
        
        # Crear la colección
        collection = Collection(name=COLLECTION_NAME, schema=schema)
        
        # Crear índice para búsqueda vectorial
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        logger.info(f"Colección {COLLECTION_NAME} creada exitosamente")
        
        return collection
            
    except Exception as e:
        logger.error(f"Error al crear la colección en Milvus: {e}")
        raise

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
            description_text = "".join(description.itertext()).strip()

        # Obtener severidad y probabilidad
        likelihood = pattern.find("capec:Likelihood_Of_Attack", namespace)
        likelihood_text = likelihood.text if likelihood is not None else "Not Specified"

        severity = pattern.find("capec:Typical_Severity", namespace)
        severity_text = severity.text if severity is not None else "Not Specified"

        # Obtener vectores de ataque
        attack_vectors = pattern.find("capec:Typical_Attack_Vectors", namespace)
        attack_vectors_text = ", ".join([v.text for v in attack_vectors]) if attack_vectors is not None else ""

        # Obtener prerrequisitos
        prerequisites = pattern.find("capec:Typical_Attack_Prerequisites", namespace)
        prerequisites_text = ", ".join([p.text for p in prerequisites]) if prerequisites is not None else ""

        # Obtener recursos requeridos
        resources = pattern.find("capec:Typical_Resources_Required", namespace)
        resources_text = ", ".join([r.text for r in resources]) if resources is not None else ""

        # Obtener mitigaciones
        mitigations = pattern.find("capec:Typical_Attack_Mitigations", namespace)
        mitigations_text = ", ".join([m.text for m in mitigations]) if mitigations is not None else ""

        # Obtener ejemplos
        examples = pattern.find("capec:Typical_Attack_Examples", namespace)
        examples_text = ", ".join([e.text for e in examples]) if examples is not None else ""

        return {
            "pattern_id": pattern_id,
            "name": pattern_name,
            "description": description_text,
            "status": pattern_status,
            "abstraction": pattern_abstraction,
            "typical_severity": severity_text,
            "typical_likelihood": likelihood_text,
            "typical_attack_vectors": attack_vectors_text,
            "typical_attack_prerequisites": prerequisites_text,
            "typical_resources_required": resources_text,
            "typical_attack_mitigations": mitigations_text,
            "typical_attack_examples": examples_text
        }
    except AttributeError as e:
        logger.error(f"Error procesando patrón: {e}")
        return None

def main():
    try:
        # Verificar si el archivo existe
        if not os.path.exists(XML_FILE_PATH):
            logger.error(f"Error: El archivo XML no se encuentra en: {XML_FILE_PATH}")
            return       

        # Crear colección en Milvus
        collection = create_milvus_collection()

        # Parsear el XML
        tree = ET.parse(XML_FILE_PATH)
        root = tree.getroot()
        attack_patterns = []

        # Definir el namespace
        namespace = (
            {"capec": root.tag.split("}")[0].strip("{")} if "}" in root.tag else ""
        )
        logger.info("Procesando archivo XML...")

        # Obtener el contenedor Attack_Patterns
        attack_patterns_container = root.find("capec:Attack_Patterns", namespace)
        if attack_patterns_container is None:
            logger.error("Error: No se encontró la estructura Attack_Patterns en el XML")
            return

        # Iterar sobre todos los patrones de ataque
        for pattern in attack_patterns_container.findall(
            "capec:Attack_Pattern", namespace
        ):
            pattern_data = extract_attack_pattern(pattern, namespace)
            if pattern_data:
                attack_patterns.append(pattern_data)

        logger.info(f"Se encontraron {len(attack_patterns)} patrones de ataque")

        # Generar embeddings
        logger.info("Generando embeddings...")
        model = SentenceTransformer("all-MiniLM-L6-v2")

        # Crear textos combinados para embedding
        texts = [
            f"{pattern['pattern_id']} {pattern['name']}: {pattern['description']}"
            for pattern in attack_patterns
        ]

        # Generar embeddings
        embeddings = model.encode(texts, convert_to_tensor=False)
        logger.info(f"Se generaron {len(embeddings)} embeddings")

        # Preparar datos para insertar en Milvus
        collection.load()

        entities = []
        for i, pattern in enumerate(attack_patterns):
            entity = pattern.copy()
            entity["embedding"] = embeddings[i].tolist()
            entities.append(entity)

        # Insertar datos en Milvus
        logger.info("Insertando datos en Milvus...")
        collection.insert(entities)
        logger.info("Datos insertados exitosamente en Milvus")

    except ET.ParseError as e:
        logger.error(f"Error al parsear el XML: {e}")
    except Exception as e:
        logger.error(f"Error inesperado: {e}")
    finally:
        # Cerrar conexión con Milvus
        connections.disconnect("default")
        logger.info("Desconectado de Milvus")

if __name__ == "__main__":
    main()
