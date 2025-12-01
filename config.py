"""
Configuración centralizada para el proyecto capec2vector.
Este módulo contiene todas las configuraciones utilizadas en el proyecto.
"""
import os
from pathlib import Path

# Rutas del proyecto
PROJECT_ROOT = Path(__file__).parent
CAPEC_DATA_DIR = PROJECT_ROOT / "capec_latest"
XML_FILE_PATH = CAPEC_DATA_DIR / "capec_v3.9.xml"

# Configuración de Milvus
MILVUS_HOST = os.environ.get("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.environ.get("MILVUS_PORT", 19530))
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "capec_patterns")

# Configuración de embeddings
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1")
EMBEDDING_DIMENSION = int(os.environ.get("EMBEDDING_DIMENSION", 768))

# Configuración alternativa para modelos más ligeros
ALTERNATIVE_MODEL = "all-MiniLM-L6-v2"
ALTERNATIVE_DIMENSION = 384

# Configuración de Ollama
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://172.16.11.224:11434")
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b")

# Configuración de la API web
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", 8000))

# Configuración de Kali API
KALI_API_PORT = int(os.environ.get("KALI_API_PORT", 5000))
KALI_API_BASE_URL = os.environ.get(
    "KALI_API_BASE_URL", f"http://172.16.11.111:{KALI_API_PORT}"
)

# Configuración de Milvus - índice vectorial
INDEX_TYPE = "IVF_FLAT"
METRIC_TYPE = "L2"
NLIST = 1024
NPROBE = 10

# Configuración de logging
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Configuración de búsqueda
DEFAULT_SEARCH_TOP_K = int(os.environ.get("DEFAULT_SEARCH_TOP_K", 5))
MAX_SEARCH_TOP_K = int(os.environ.get("MAX_SEARCH_TOP_K", 100))

# Configuración de caché
ENABLE_CACHE = os.environ.get("ENABLE_CACHE", "true").lower() == "true"
CACHE_DIR = PROJECT_ROOT / ".cache"
CACHE_TTL = int(os.environ.get("CACHE_TTL", 3600))  # 1 hora por defecto

# Herramientas Kali disponibles
KALI_TOOLS = {
    "nmap": "/api/tools/nmap",
    "gobuster": "/api/tools/gobuster",
    "dirb": "/api/tools/dirb",
    "nikto": "/api/tools/nikto",
    "sqlmap": "/api/tools/sqlmap",
    "metasploit": "/api/tools/metasploit",
    "hydra": "/api/tools/hydra",
    "john": "/api/tools/john",
    "wpscan": "/api/tools/wpscan",
    "enum4linux": "/api/tools/enum4linux",
}

# Configuración del servidor MCP (WebSocket)
MCP_HOST = os.environ.get("MCP_HOST", "localhost")
MCP_PORT = int(os.environ.get("MCP_PORT", 8765))

# Límites de campos VARCHAR en Milvus
MAX_LENGTHS = {
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
    "outcomes": 5000,
}


def get_config_summary():
    """Retorna un resumen de la configuración actual"""
    return {
        "milvus": {
            "host": MILVUS_HOST,
            "port": MILVUS_PORT,
            "collection": COLLECTION_NAME,
        },
        "embeddings": {
            "model": EMBEDDING_MODEL,
            "dimension": EMBEDDING_DIMENSION,
        },
        "ollama": {
            "host": OLLAMA_HOST,
            "model": DEFAULT_OLLAMA_MODEL,
        },
        "api": {
            "host": API_HOST,
            "port": API_PORT,
        },
        "cache": {
            "enabled": ENABLE_CACHE,
            "ttl": CACHE_TTL,
        },
    }


if __name__ == "__main__":
    """Imprime la configuración actual cuando se ejecuta directamente"""
    import json
    print(json.dumps(get_config_summary(), indent=2))
