import xml.etree.ElementTree as ET
import os
from sentence_transformers import SentenceTransformer


# Definir la ruta del archivo como una constante
XML_FILE_PATH = os.path.join('capec_latest', 'capec_v3.9.xml')

def extract_attack_pattern(pattern, namespace):
    """Extrae la información relevante de un patrón de ataque"""
    try:
        # Obtener atributos básicos
        pattern_id = pattern.get('ID')
        pattern_name = pattern.get('Name')
        pattern_status = pattern.get('Status')
        pattern_abstraction = pattern.get('Abstraction')
        
        # Obtener elementos de texto
        description = pattern.find('capec:Description', namespace)
        description_text = description.text if description is not None and description.text else ""
        if description is not None and len(description) > 0:
            # Si hay elementos hijos, obtener el texto de todos los hijos
            description_text = ''.join(description.itertext()).strip()
            
        # Obtener severidad y probabilidad
        likelihood = pattern.find('capec:Likelihood_Of_Attack', namespace)
        likelihood_text = likelihood.text if likelihood is not None else "Not Specified"
        
        severity = pattern.find('capec:Typical_Severity', namespace)
        severity_text = severity.text if severity is not None else "Not Specified"
        
        return {
            'id': pattern_id,
            'name': pattern_name,
            'status': pattern_status,
            'abstraction': pattern_abstraction,
            'description': description_text,
            'likelihood': likelihood_text,
            'severity': severity_text
        }
    except AttributeError as e:
        print(f"Error procesando patrón: {e}")
        return None

def main():
    try:
        # Verificar si el archivo existe
        if not os.path.exists(XML_FILE_PATH):
            print(f"Error: El archivo XML no se encuentra en: {XML_FILE_PATH}")
            return
            
        # Parsear el XML
        tree = ET.parse(XML_FILE_PATH)
        root = tree.getroot()
        attack_patterns = []
        
        # Definir el namespace
        namespace = {'capec': root.tag.split('}')[0].strip('{')} if '}' in root.tag else ''
        print("Procesando archivo XML...")
        
        # Obtener el contenedor Attack_Patterns
        attack_patterns_container = root.find('capec:Attack_Patterns', namespace)
        if attack_patterns_container is None:
            print("Error: No se encontró la estructura Attack_Patterns en el XML")
            return
            
        # Iterar sobre todos los patrones de ataque dentro del contenedor
        for pattern in attack_patterns_container.findall('capec:Attack_Pattern', namespace):
            pattern_data = extract_attack_pattern(pattern, namespace)
            if pattern_data:
                attack_patterns.append(pattern_data)
        
        print(f"\nSe encontraron {len(attack_patterns)} patrones de ataque")
        
        # Imprimir algunos ejemplos y generar embeddings
        print("\nGenerando embeddings...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Crear textos combinados para embedding
        texts = [f"{pattern['id']} {pattern['name']}: {pattern['description']}" 
                for pattern in attack_patterns]
        
        # Generar embeddings
        embeddings = model.encode(texts, convert_to_tensor=False)
        print(f"\nSe generaron {len(embeddings)} embeddings")
        print(f"Dimensión de cada embedding: {embeddings.shape[1]}")
        
        # Imprimir ejemplos de patrones
        for pattern in attack_patterns[:5]:
            print(f"\nID: {pattern['id']}")
            print(f"Nombre: {pattern['name']}")
            print(f"Estado: {pattern['status']}")
            print(f"Abstracción: {pattern['abstraction']}")
            print(f"Probabilidad: {pattern['likelihood']}")
            print(f"Severidad: {pattern['severity']}")
            print("Descripción:", pattern['description'][:200] + "..." if len(pattern['description']) > 200 else pattern['description'])
            print("-" * 80)

    except ET.ParseError as e:
        print(f"Error al parsear el XML: {e}")
    except Exception as e:
        print(f"Error inesperado: {e}")

if __name__ == "__main__":
    main()