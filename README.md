# CAPEC2Vector

Una herramienta para convertir entradas CAPEC (Common Attack Pattern Enumeration and Classification) en vectores de embeddings para aplicaciones de aprendizaje automático en ciberseguridad.

## Descripción

Este proyecto proporciona un pipeline para procesar descripciones de patrones de ataque CAPEC y convertirlas en representaciones vectoriales numéricas utilizando técnicas avanzadas de PLN. Estos vectores pueden utilizarse para diversas tareas de aprendizaje automático como clasificación de patrones de ataque, análisis de similitud y detección de amenazas.

## Características

- Análisis de archivos XML de CAPEC
- Procesamiento y limpieza de descripciones de patrones de ataque
- Generación de embeddings vectoriales a partir de descripciones textuales
- Soporte para múltiples modelos de embeddings
- Procesamiento y almacenamiento eficiente de datos

## Instalación

```bash
git clone https://github.com/yourusername/capec2vector.git
cd capec2vector
pip install -r requirements.txt
```

## Uso

1. Coloca tu archivo XML de CAPEC en el directorio data
2. Ejecuta el script principal:

```bash
python main.py --input data/capec.xml --output vectors/
```

## Requisitos

- Python 3.8+
- Los paquetes requeridos están listados en requirements.txt

## Licencia

[Licencia MIT](LICENSE)

## Contribuciones

¡Las contribuciones son bienvenidas! No dudes en enviar un Pull Request.

## Base de Datos Milvus

Milvus es un sistema de base de datos vectorial de código abierto diseñado para el procesamiento de datos a gran escala y búsqueda de similitud. Características principales:


# CAPEC2Vector

Repositorio para extraer descripciones del catálogo CAPEC, generar embeddings a partir de esos textos y exponer una API web que integra Milvus (almacenamiento vectorial) con un LLM (Ollama) para consultas enriquecidas.

Este README se ha actualizado para reflejar la estructura real del proyecto y los scripts disponibles.

## Qué incluye este repositorio

- `embeddings.py`: extracción y limpieza del archivo XML `capec_latest/capec_v3.9.xml`, generación de embeddings (usa `nomic-ai/nomic-embed-text-v1` / `sentence-transformers`) y creación de la colección en Milvus.
- `pipeline.py`: pipeline ejemplo que muestra la integración (parseo XML -> embeddings -> Milvus -> consulta -> Ollama).
- `ollama_milvus_bridge.py`: aplicación FastAPI que expone endpoints para buscar patrones CAPEC en Milvus y generar respuestas con Ollama. También sirve una UI estática en `templates/index.html`.
- `ollama_adapter.py`: adaptador orientado a integrar Ollama con un API de herramientas tipo "Kali" (p. ej. ejecutar herramientas remotas vía API).
- `mcp_server.py`: servidor WebSocket (MCP) que permite ejecutar comandos remotos y devolver resultados (utilizado por integraciones en tiempo real).
- `requirements.txt`: paquetes Python necesarios.
- `docker-compose.yml`: fichero para levantar Milvus, etcd y MinIO (ya incluido en el repo).
- `capec_latest/`: carpeta con el XML de CAPEC (`capec_v3.9.xml`) ya presente en el repositorio.

## Requisitos

- Python 3.8+
- Docker & Docker Compose (opcional, para levantar Milvus/etcd/MinIO localmente)
- Paquetes Python listados en `requirements.txt`.

Instalación rápida (virtualenv recomendado):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Nota: algunos modelos de embedding (y PyTorch) pueden requerir GPU o ajustes de versión.

## Infraestructura (Milvus + MinIO + etcd)

Se incluye un `docker-compose.yml` preparado para levantar Milvus (standalone), MinIO y etcd. Para ejecutar:

```bash
# Desde la raíz del repositorio
Abre tu navegador y visita `http://localhost:8000`
```

Verifica los servicios:

```bash

```

Si trabajas con Milvus remoto, ajusta las variables de entorno `MILVUS_HOST` y `MILVUS_PORT` antes de ejecutar los scripts.

## Flujo principal y comandos de uso

1) Crear la colección en Milvus e importar los patrones CAPEC (generar embeddings):

```bash
# Ejecuta el script principal de embeddings (crea colección y carga datos desde capec_latest/capec_v3.9.xml)
python embeddings.py
```

El script `embeddings.py` hace:
- Parse del XML `capec_latest/capec_v3.9.xml`.
- Limpieza y normalización de campos relevantes.
- Generación de textos enriquecidos por patrón y cálculo de embeddings con `nomic-ai/nomic-embed-text-v1`.
- Creación de la colección `capec_patterns` en Milvus y carga de vectores + metadatos.

2) Iniciar la API web que integra Milvus con Ollama (FastAPI):

```bash
# Recomendado: ejecutar con uvicorn
uvicorn ollama_milvus_bridge:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints principales:
- `POST /search` : búsqueda semántica en Milvus (payload: {"query": "...", "top_k": 10}).
- `POST /ollama/query` : búsqueda + generación de respuesta con Ollama (devuelve `answer` y `relevant_patterns`).
- `/` : interfaz web (usa `templates/index.html`).

Variables de configuración relevantes (pueden definirse como variables de entorno):
- `MILVUS_HOST` (default: localhost)
- `MILVUS_PORT` (default: 19530)
- `COLLECTION_NAME` (usado por los scripts; default: capec_patterns)
- `OLLAMA_HOST` (URL del servicio Ollama)
- `API_PORT` / `KALI_API_BASE_URL` (usados por `ollama_adapter.py` / `ollama_milvus_bridge.py` si integran herramientas externas)

3) Adaptador Ollama -> herramientas (opcional):

```bash
# Lanza el adaptador interactivo (requiere Ollama local accesible y un API de "Kali" remoto si se va a ejecutar herramientas)
python ollama_adapter.py
```

4) Servidor MCP (WebSocket) para ejecución remota de comandos:

```bash
python mcp_server.py
```

## Notas de seguridad y uso responsable

- Este repositorio contiene componentes que pueden interactuar con herramientas de seguridad ofensivas (por ejemplo, integraciones orientadas a Kali). Úsalos únicamente en entornos controlados y con permiso explícito del propietario del objetivo.
- Asegúrate de no exponer Ollama o el API de ejecución de comandos a redes públicas sin autenticación.

## Desarrollo y ajuste de modelos

- El proyecto usa `nomic-ai/nomic-embed-text-v1` por defecto para embeddings. Puedes cambiar el modelo en `embeddings.py` o `ollama_milvus_bridge.py` (función que inicializa `SentenceTransformer`).
- `DIMENSION` en `embeddings.py` y la configuración de índice en Milvus deben concordar con la dimensión del embedding elegido.

## Archivos clave y su propósito

- `embeddings.py` — extracción, limpieza, generación de embeddings y carga en Milvus.
- `pipeline.py` — ejemplo de pipeline completo y utilidades auxiliares (parseo, búsqueda, respuesta con Ollama).
- `ollama_milvus_bridge.py` — FastAPI que expone la funcionalidad de búsqueda y generación por LLM.
- `ollama_adapter.py` — adaptador para integrar Ollama con APIs de herramientas (Kali).
- `mcp_server.py` — servidor WebSocket para ejecutar comandos remotos.

## Contribuciones

Pull requests y issues son bienvenidos. Para cambios importantes, abre una issue primero describiendo la propuesta.

## Licencia

Licencia MIT (si procede). Revisa el archivo `LICENSE` si existe.

---

Si quieres que adapte este README para incluir pasos reproducibles en tu entorno (por ejemplo, configuración de variables de entorno, instrucciones para usar Ollama localmente, o pasos para reproducir la carga de datos con un subconjunto del XML), dime y lo agrego.
