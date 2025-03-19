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

## Uso

1. Iniciar el servidor:
```bash
python ollama_milvus_bridge.py
```

2. Acceder a la interfaz web:
