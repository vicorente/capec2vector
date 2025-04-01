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
DIMENSION = 768  # Dimensión del embedding de nomic

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
            FieldSchema(name="pattern_id", dtype=DataType.VARCHAR, max_length=20),  # Ajustado según el tamaño típico de un ID
            FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=200),  # Ajustado según el tamaño máximo observado en nombres
            FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=10000),  # Ajustado para descripciones largas
            FieldSchema(name="status", dtype=DataType.VARCHAR, max_length=20),  # Ajustado según los valores posibles de estado
            FieldSchema(name="abstraction", dtype=DataType.VARCHAR, max_length=20),  # Ajustado según los valores posibles de abstracción
            FieldSchema(name="Typical_Severity", dtype=DataType.VARCHAR, max_length=30),  # Ajustado según los valores posibles de severidad
            FieldSchema(name="Likelihood_Of_Attack", dtype=DataType.VARCHAR, max_length=30),  # Ajustado según los valores posibles de probabilidad
            FieldSchema(name="Prerequisites", dtype=DataType.VARCHAR, max_length=2000),  # Ajustado para listas largas de prerrequisitos
            FieldSchema(name="Resources_Required", dtype=DataType.VARCHAR, max_length=2000),  # Ajustado para listas largas de recursos
            FieldSchema(name="Mitigations", dtype=DataType.VARCHAR, max_length=2000),  # Ajustado para listas largas de mitigaciones
            FieldSchema(name="Example_Instances", dtype=DataType.VARCHAR, max_length=5000),  # Ajustado para listas largas de ejemplos
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)  # Dimensión fija para embeddings
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

        # Obtener elementos usando los nombres correctos del XML
        prerequisites = pattern.find("capec:Prerequisites", namespace)
        prerequisites_text = ", ".join([p.text for p in prerequisites]) if prerequisites is not None and len(prerequisites) > 0 else ""

        resources = pattern.find("capec:Resources_Required", namespace)
        resources_text = ", ".join([r.text for r in resources]) if resources is not None and len(resources) > 0 else ""

        mitigations = pattern.find("capec:Mitigations", namespace)
        mitigations_text = ", ".join([m.text for m in mitigations]) if mitigations is not None and len(mitigations) > 0 else ""

        examples = pattern.find("capec:Example_Instances", namespace)
        examples_text = ", ".join([e.text for e in examples]) if examples is not None and len(examples) > 0 else ""

        return {
            "pattern_id": "CAPEC-" + pattern_id,
            "name": pattern_name,
            "description": description_text,
            "status": pattern_status,
            "abstraction": pattern_abstraction,
            "Typical_Severity": severity_text,
            "Likelihood_Of_Attack": likelihood_text,
            "Prerequisites": prerequisites_text,
            "Resources_Required": resources_text,
            "Mitigations": mitigations_text,
            "Example_Instances": examples_text
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
        model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

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
