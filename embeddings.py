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
        # Función auxiliar para extraer texto directo de elementos
        def get_direct_text(element):
            """Obtiene solo el texto directo de un elemento, sin incluir el texto de sus hijos"""
            text = ""
            if element is not None:
                # Obtener solo el texto directo del elemento
                if element.text:
                    text += element.text.strip()
                # No incluir el texto de los elementos hijos
                for child in element:
                    if child.tail:
                        text += child.tail.strip()
            return text

        def get_element_text(element, xpath, join_char=", "):
            """Extrae texto de elementos considerando solo el texto directo"""
            items = element.findall(xpath, namespace)
            if items:
                return join_char.join(
                    get_direct_text(item)
                    for item in items 
                    if get_direct_text(item)
                )
            return ""

        # Función auxiliar para extraer steps/phases
        def extract_execution_flow(element, step_type):
            steps = element.findall(f".//capec:{step_type}", namespace)
            flow = []
            for step in steps:
                phase = step.get("Phase")
                number = step.get("Number", "")
                text = get_direct_text(step)
                flow.append(f"Phase {phase} - Step {number}: {text}")
            return "\n".join(flow)

        # Extraer descripción de forma limpia
        description_elem = pattern.find(".//capec:Description", namespace)
        description = get_direct_text(description_elem) if description_elem is not None else ""

        # Datos básicos con la descripción limpia
        pattern_data = {
            "pattern_id": "CAPEC-" + pattern.get("ID", ""),
            "name": pattern.get("Name", ""),
            "status": pattern.get("Status", ""),
            "abstraction": pattern.get("Abstraction", ""),
            "description": description,  # Usar la descripción limpia
            
            # Descripción y resumen
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
    """Crea un texto enriquecido para generar el embedding incluyendo todos los campos del patrón"""
    # Primero verificamos que el patrón tenga todas las claves necesarias
    required_fields = [
        'pattern_id', 'name', 'status', 'abstraction', 'description', 
        'summary', 'alternate_terms', 'submission_date', 'submission_name',
        'submission_organization', 'typical_severity', 'likelihood_of_attack',
        'prerequisites', 'skills_required', 'resources_required', 'indicators',
        'consequences', 'mitigations', 'example_instances', 'notes',
        'related_attack_patterns', 'related_weaknesses', 'taxonomy_mappings',
        'execution_flow', 'attack_steps', 'outcomes'
    ]
    
    # Asegurarnos de que todos los campos existan, si no, usar valor vacío
    safe_pattern = {field: pattern.get(field, '') for field in required_fields}
    
    sections = [
        # Información básica
        f"CAPEC-{safe_pattern['pattern_id']}",
        f"Name: {safe_pattern['name']}",
        f"Status: {safe_pattern['status']}",
        f"Abstraction: {safe_pattern['abstraction']}",
        
        # Descripción y resumen
        f"Description: {safe_pattern['description']}",
        f"Summary: {safe_pattern['summary']}",
        f"Alternate Terms: {safe_pattern['alternate_terms']}",
        
        # Metadata de sumisión
        f"Submission Date: {safe_pattern['submission_date']}",
        f"Submission Name: {safe_pattern['submission_name']}",
        f"Submission Organization: {safe_pattern['submission_organization']}",
        
        # Evaluación de riesgo
        f"Typical Severity: {safe_pattern['typical_severity']}",
        f"Likelihood of Attack: {safe_pattern['likelihood_of_attack']}",
        
        # Detalles técnicos
        f"Prerequisites: {safe_pattern['prerequisites']}",
        f"Skills Required: {safe_pattern['skills_required']}",
        f"Resources Required: {safe_pattern['resources_required']}",
        f"Indicators: {safe_pattern['indicators']}",
        
        # Consecuencias y mitigación
        f"Consequences: {safe_pattern['consequences']}",
        f"Mitigations: {safe_pattern['mitigations']}",
        
        # Ejemplos y notas
        f"Example Instances: {safe_pattern['example_instances']}",
        f"Notes: {safe_pattern['notes']}",
        
        # Relaciones
        f"Related Attack Patterns: {safe_pattern['related_attack_patterns']}",
        f"Related Weaknesses: {safe_pattern['related_weaknesses']}",
        f"Taxonomy Mappings: {safe_pattern['taxonomy_mappings']}",
        
        # Flujo de ejecución
        f"Execution Flow: {safe_pattern['execution_flow']}",
        f"Attack Steps: {safe_pattern['attack_steps']}",
        f"Outcomes: {safe_pattern['outcomes']}"
    ]
    
    # Filtrar secciones vacías y unir con separador
    filtered_sections = []
    for section in sections:
        parts = section.split(': ', 1)  # Dividir solo en la primera aparición de ': '
        if len(parts) > 1 and parts[1].strip():
            filtered_sections.append(section)
    
    return " | ".join(filtered_sections) if filtered_sections else "No data available"

def clean_and_validate_data(pattern_data):
    """Limpia y valida los datos antes de la inserción en Milvus"""
    try:
        # Función auxiliar para truncar texto
        def truncate_text(text, max_length):
            return str(text)[:max_length] if text else ""

        # Definir límites máximos según el esquema
        max_lengths = {
            "pattern_id": 20,
            "name": 200,
            "description": 10000,
            "status": 20,
            "abstraction": 20,
            "summary": 5000,
            "alternate_terms": 2000,
            "submission_date": 20,
            "submission_name": 100,
            "submission_organization": 100,
            "typical_severity": 30,
            "likelihood_of_attack": 30,
            "prerequisites": 5000,
            "skills_required": 5000,
            "resources_required": 5000,
            "indicators": 5000,
            "consequences": 5000,
            "mitigations": 10000,
            "example_instances": 10000,
            "notes": 5000,
            "related_attack_patterns": 5000,
            "related_weaknesses": 5000,
            "taxonomy_mappings": 5000,
            "execution_flow": 10000,
            "attack_steps": 10000,
            "outcomes": 5000
        }

        # Limpiar y validar cada campo
        clean_data = {}
        for field, value in pattern_data.items():
            if field == "embedding":
                clean_data[field] = value
                continue
                
            max_length = max_lengths.get(field)
            if max_length:
                clean_data[field] = truncate_text(value, max_length)
            else:
                clean_data[field] = str(value) if value else ""

        return clean_data
    except Exception as e:
        logger.error(f"Error en la limpieza y validación de datos: {e}")
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
        
        logger.info("Iniciando procesamiento del catálogo CAPEC...")
        logger.info(f"Versión del catálogo: {root.get('Version')}")
        logger.info(f"Fecha del catálogo: {root.get('Date')}")

        # Obtener el contenedor Attack_Patterns
        attack_patterns_container = root.find("capec:Attack_Patterns", namespace)
        if (attack_patterns_container is None):
            logger.error("Error: No se encontró la estructura Attack_Patterns en el XML")
            return

        # Iterar sobre todos los patrones de ataque
        for pattern in attack_patterns_container.findall(
            "capec:Attack_Pattern", namespace
        ):
            pattern_data = extract_attack_pattern(pattern, namespace)
            if (pattern_data):
                attack_patterns.append(pattern_data)

        logger.info(f"Se encontraron {len(attack_patterns)} patrones de ataque")        
        

        # Generar embeddings
        logger.info("Generando embeddings...")
        model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

        # Crear textos combinados para embedding y filtrar patrones vacíos
        valid_patterns = []
        valid_texts = []
        
        for pattern in attack_patterns:
            text = create_pattern_embedding_text(pattern)
            if text != "No data available":
                valid_patterns.append(pattern)
                valid_texts.append(text)
        
        if not valid_texts:
            logger.error("No se encontraron patrones válidos para generar embeddings")
            return
            
        # Generar embeddings
        logger.info(f"Generando embeddings para {len(valid_texts)} patrones válidos...")
        embeddings = model.encode(valid_texts, convert_to_tensor=False)
        logger.info(f"Se generaron {len(embeddings)} embeddings")

        # Preparar datos para insertar en Milvus
        collection.load()

        entities = []
        batch_size = 1000  # Procesar en lotes para mejor manejo de memoria
        current_batch = []

        for i, pattern in enumerate(valid_patterns):
            entity = pattern.copy()
            entity["embedding"] = embeddings[i].tolist()
            
            # Limpiar y validar los datos
            clean_entity = clean_and_validate_data(entity)
            if clean_entity:
                current_batch.append(clean_entity)
            
            # Insertar cuando el lote esté completo o sea el último elemento
            if len(current_batch) >= batch_size or i == len(valid_patterns) - 1:
                try:
                    mr = collection.insert(current_batch)
                    logger.info(f"Insertados {len(current_batch)} registros. IDs: {mr.primary_keys}")
                    current_batch = []
                except Exception as e:
                    logger.error(f"Error al insertar lote en Milvus: {e}")
                    continue

        # Asegurar que los datos estén persistidos
        collection.flush()
        logger.info(f"Total de patrones insertados exitosamente en Milvus: {collection.num_entities}")

    except ET.ParseError as e:
        logger.error(f"Error al parsear el XML: {e}")
    except Exception as e:
        logger.error(f"Error inesperado: {e}")
    finally:
        # Cerrar conexión con Milvus
        collection.release()
        connections.disconnect("default")
        logger.info("Desconectado de Milvus")

if __name__ == "__main__":
    main()
