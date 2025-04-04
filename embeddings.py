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
    """Crea la colección en Milvus con todos los campos del CAPEC"""
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
            # Campos básicos existentes
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="pattern_id", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=10000),
            FieldSchema(name="status", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="abstraction", dtype=DataType.VARCHAR, max_length=20),
            
            # Campos adicionales para detalles completos
            FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="alternate_terms", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="submission_date", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="submission_name", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="submission_organization", dtype=DataType.VARCHAR, max_length=100),
            
            # Campos de evaluación de riesgo
            FieldSchema(name="typical_severity", dtype=DataType.VARCHAR, max_length=30),
            FieldSchema(name="likelihood_of_attack", dtype=DataType.VARCHAR, max_length=30),
            
            # Campos técnicos
            FieldSchema(name="prerequisites", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="skills_required", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="resources_required", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="indicators", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="consequences", dtype=DataType.VARCHAR, max_length=5000),
            
            # Campos de mitigación y ejemplos
            FieldSchema(name="mitigations", dtype=DataType.VARCHAR, max_length=10000),
            FieldSchema(name="example_instances", dtype=DataType.VARCHAR, max_length=10000),
            FieldSchema(name="notes", dtype=DataType.VARCHAR, max_length=5000),
            
            # Campos de relaciones
            FieldSchema(name="related_attack_patterns", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="related_weaknesses", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="taxonomy_mappings", dtype=DataType.VARCHAR, max_length=5000),
            
            # Campos de ejecución
            FieldSchema(name="execution_flow", dtype=DataType.VARCHAR, max_length=10000),
            FieldSchema(name="attack_steps", dtype=DataType.VARCHAR, max_length=10000),
            FieldSchema(name="outcomes", dtype=DataType.VARCHAR, max_length=5000),
            
            # Vector de embedding
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
    """Extrae toda la información disponible de un patrón de ataque"""
    try:
        # Función auxiliar para extraer texto de elementos
        def get_element_text(element, xpath, join_char=", "):
            items = element.findall(xpath, namespace)
            if items:
                return join_char.join(item.text.strip() for item in items if item.text)
            return ""

        # Función auxiliar para extraer steps/phases
        def extract_execution_flow(element, step_type):
            steps = element.findall(f".//capec:{step_type}", namespace)
            flow = []
            for step in steps:
                phase = step.get("Phase")
                number = step.get("Number", "")
                text = "".join(step.itertext()).strip()
                flow.append(f"Phase {phase} - Step {number}: {text}")
            return "\n".join(flow)

        # Datos básicos
        pattern_data = {
            "pattern_id": "CAPEC-" + pattern.get("ID", ""),
            "name": pattern.get("Name", ""),
            "status": pattern.get("Status", ""),
            "abstraction": pattern.get("Abstraction", ""),
            
            # Descripción y resumen
            "description": get_element_text(pattern, ".//capec:Description", "\n"),
            "summary": get_element_text(pattern, ".//capec:Summary", "\n"),
            "alternate_terms": get_element_text(pattern, ".//capec:Alternate_Terms//capec:Term"),
            
            # Metadata de sumisión
            "submission_date": pattern.get("Submission_Date", ""),
            "submission_name": pattern.get("Submission_Name", ""),
            "submission_organization": pattern.get("Submission_Organization", ""),
            
            # Evaluación de riesgo
            "typical_severity": get_element_text(pattern, ".//capec:Typical_Severity"),
            "likelihood_of_attack": get_element_text(pattern, ".//capec:Likelihood_Of_Attack"),
            
            # Detalles técnicos
            "prerequisites": get_element_text(pattern, ".//capec:Prerequisites//capec:Prerequisite", "\n"),
            "skills_required": get_element_text(pattern, ".//capec:Skills_Required//capec:Skill", "\n"),
            "resources_required": get_element_text(pattern, ".//capec:Resources_Required//capec:Resource", "\n"),
            "indicators": get_element_text(pattern, ".//capec:Indicators//capec:Indicator", "\n"),
            
            # Consecuencias
            "consequences": get_element_text(pattern, ".//capec:Consequences//capec:Consequence", "\n"),
            
            # Mitigación y ejemplos
            "mitigations": get_element_text(pattern, ".//capec:Mitigations//capec:Mitigation", "\n"),
            "example_instances": get_element_text(pattern, ".//capec:Example_Instances//capec:Example", "\n"),
            "notes": get_element_text(pattern, ".//capec:Notes//capec:Note", "\n"),
            
            # Relaciones
            "related_attack_patterns": ", ".join([
                f"CAPEC-{ref.get('CAPEC_ID')}"
                for ref in pattern.findall(".//capec:Related_Attack_Patterns//capec:Related_Attack_Pattern", namespace)
            ]),
            
            "related_weaknesses": ", ".join([
                f"CWE-{ref.get('CWE_ID')}"
                for ref in pattern.findall(".//capec:Related_Weaknesses//capec:Related_Weakness", namespace)
            ]),
            
            # Mapeos de taxonomía
            "taxonomy_mappings": ", ".join([
                f"{mapping.get('Taxonomy_Name')}-{mapping.get('Entry_ID')}"
                for mapping in pattern.findall(".//capec:Taxonomy_Mappings//capec:Taxonomy_Mapping", namespace)
            ]),
            
            # Flujo de ejecución
            "execution_flow": extract_execution_flow(pattern, "Attack_Step"),
            "attack_steps": extract_execution_flow(pattern, "Technique"),
            "outcomes": get_element_text(pattern, ".//capec:Outcomes//capec:Outcome", "\n")
        }

        return pattern_data

    except Exception as e:
        logger.error(f"Error procesando patrón {pattern.get('ID')}: {e}")
        return None

def create_pattern_embedding_text(pattern):
    """Crea un texto enriquecido para generar el embedding"""
    sections = [
        f"CAPEC-{pattern['pattern_id']}",
        f"Name: {pattern['name']}",
        f"Description: {pattern['description']}",
        f"Summary: {pattern['summary']}",
        f"Prerequisites: {pattern['prerequisites']}",
        f"Attack Steps: {pattern['attack_steps']}",
        f"Consequences: {pattern['consequences']}",
        f"Mitigations: {pattern['mitigations']}",
        f"Related Weaknesses: {pattern['related_weaknesses']}",
        f"Related Patterns: {pattern['related_attack_patterns']}"
    ]
    return " | ".join(filter(None, sections))

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
        
        logger.info("Iniciando procesamiento del catálogo CAPEC...")
        logger.info(f"Versión del catálogo: {root.get('Version')}")
        logger.info(f"Fecha del catálogo: {root.get('Date')}")

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
        
        logger.info(f"Procesados {len(attack_patterns)} patrones de ataque")
        logger.info("Desglose por abstracción:")
        abstractions = {}
        for pattern in attack_patterns:
            abs_level = pattern['abstraction'] or 'Unspecified'
            abstractions[abs_level] = abstractions.get(abs_level, 0) + 1
        for abs_level, count in abstractions.items():
            logger.info(f"  - {abs_level}: {count} patrones")

        # Generar embeddings
        logger.info("Generando embeddings...")
        model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

        # Crear textos combinados para embedding
        texts = [create_pattern_embedding_text(pattern) for pattern in attack_patterns]

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
