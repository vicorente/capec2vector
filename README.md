# CAPEC2Vector

Sistema de búsqueda semántica de patrones de ataque CAPEC (Common Attack Pattern Enumeration and Classification) utilizando embeddings vectoriales y LLMs.

## Descripción

Este proyecto implementa un sistema de búsqueda semántica que permite:
- Convertir patrones de ataque CAPEC a embeddings vectoriales
- Almacenar y buscar patrones en una base de datos vectorial (Milvus)
- Realizar consultas en lenguaje natural
- Generar respuestas detalladas utilizando LLMs (Ollama)

## Características

- 🔍 Búsqueda semántica de patrones CAPEC
- 🤖 Integración con modelos de lenguaje a través de Ollama
- 📊 Almacenamiento vectorial con Milvus
- 🌐 API REST con FastAPI
- ⚡ Streaming de respuestas en tiempo real
- 🎨 Interfaz web interactiva

## Tecnologías

- **Backend:**
  - FastAPI
  - Milvus
  - Sentence Transformers
  - Ollama
  - Python 3.x

- **Frontend:**
  - HTML5
  - TailwindCSS
  - JavaScript

## Requisitos

- Python 3.x
- Milvus Server
- Ollama Server
- Dependencias de Python (ver `requirements.txt`)

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/tu-usuario/CAPEC2Vector.git
cd CAPEC2Vector
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

3. Configurar variables de entorno:
```bash
export MILVUS_HOST=localhost
export MILVUS_PORT=19530
export OLLAMA_HOST=http://localhost:11434
```
## Milvus Database

Milvus es un sistema de base de datos vectorial de código abierto diseñado para el procesamiento de datos a gran escala y búsqueda de similitud. Características principales:

- 🚀 Alto rendimiento en búsqueda de similitud
- 📊 Optimizado para embeddings y datos vectoriales
- 🔄 Escalabilidad horizontal
- 🔍 Búsqueda aproximada de vecinos más cercanos (ANN)
- 🛡️ Consistencia ACID

### Configuración con Docker Compose

1. Asegúrate de tener Docker y Docker Compose instalados:
```bash
sudo apt update
sudo apt install docker.io docker-compose
```

2. Crea un archivo `docker-compose.yml`:
```bash
# filepath: /home/ciberlab/IA/sacia/capec2vector/docker-compose.yml
version: '3.5'

services:
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/etcd:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd

  minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/minio:/minio_data
    command: minio server /minio_data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  standalone:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.3.3
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/milvus:/var/lib/milvus
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - "etcd"
      - "minio"

networks:
  default:
    name: milvus
```

3. Inicia los servicios:
```bash
docker-compose up -d
```

4. Verifica el estado de los contenedores:
```bash
docker-compose ps
```

5. Para detener los servicios:
```bash
docker-compose down
```

### Verificación de Milvus

Para comprobar que Milvus está funcionando correctamente:

```python
from pymilvus import connections

# Conectar a Milvus
connections.connect(host='localhost', port='19530')

# Si no hay errores, la conexión fue exitosa
print("Conexión exitosa a Milvus")
```

### Mantenimiento

- Logs de los contenedores:
```bash
docker-compose logs -f
```

- Reiniciar servicios:
```bash
docker-compose restart
```

- Limpiar volúmenes (¡PRECAUCIÓN! Elimina todos los datos):
```bash
docker-compose down -v
## Uso

1. Iniciar el servidor:
```bash
python ollama_milvus_bridge.py
```

2. Acceder a la interfaz web:
