# CAPEC2Vector

Una herramienta para convertir entradas CAPEC (Common Attack Pattern Enumeration and Classification) en vectores de embeddings para aplicaciones de aprendizaje automático en ciberseguridad.

## 🎯 Descripción

Este proyecto proporciona un pipeline completo para procesar descripciones de patrones de ataque CAPEC y convertirlas en representaciones vectoriales numéricas utilizando técnicas avanzadas de PLN. Estos vectores se almacenan en Milvus y pueden utilizarse para diversas tareas como búsqueda semántica, clasificación de patrones de ataque, análisis de similitud y detección de amenazas.

## ✨ Características Principales

### Core Features
- 🔍 **Búsqueda Semántica**: Encuentra patrones CAPEC usando lenguaje natural
- 🤖 **Integración con LLM**: Respuestas contextualizadas usando Ollama
- 🗄️ **Base de Datos Vectorial**: Almacenamiento eficiente en Milvus
- 📊 **Interfaz Web**: UI interactiva para explorar patrones
- 🔌 **API REST**: FastAPI con endpoints completos

### Nuevas Características 🆕
- ⚙️ **Configuración Centralizada**: Gestión unificada de configuraciones
- ✅ **Tests Automatizados**: Suite de tests unitarios
- 💾 **Sistema de Caché**: Cache inteligente de embeddings con TTL
- 🛡️ **Validadores**: Sanitización y validación robusta de datos
- 🔧 **Script de Setup**: Verificación automática del entorno
- 💼 **Backup/Restauración**: Herramientas para Milvus
- 📈 **Análisis y Estadísticas**: Generación de reportes completos
- 📚 **Documentación Completa**: Guías y ejemplos de uso

## 🚀 Inicio Rápido

### Opción 1: Setup Automático (Recomendado)

```bash
# Clonar repositorio
git clone https://github.com/vicorente/capec2vector.git
cd capec2vector

# Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Verificar setup
python setup.py
```

### Opción 2: Setup Manual

Ver [QUICKSTART.md](QUICKSTART.md) para instrucciones detalladas.

## 📖 Documentación

- 📘 [QUICKSTART.md](QUICKSTART.md) - Guía de inicio rápido
- 📗 [FEATURES.md](FEATURES.md) - Documentación de características
- 📙 [API_DOCS.md](API_DOCS.md) - Documentación de la API REST
- 📕 [examples/](examples/) - Ejemplos de uso prácticos

## 💻 Uso Básico

### 1. Iniciar servicios

```bash
# Levantar Milvus
docker-compose up -d

# Generar embeddings (primera vez)
python embeddings.py

# Iniciar API
uvicorn ollama_milvus_bridge:app --reload
```

### 2. Usar la interfaz web

Abre tu navegador en: http://localhost:8000

### 3. Usar la API desde Python

```python
import requests

# Buscar patrones
response = requests.post(
    "http://localhost:8000/search",
    json={"query": "SQL injection attacks", "top_k": 5}
)

results = response.json()
for pattern in results["results"]:
    print(f"{pattern['pattern_id']}: {pattern['name']}")
```

Ver más ejemplos en [examples/](examples/)

## 🛠️ Herramientas Incluidas

### Scripts de Utilidad

```bash
# Verificar setup
python setup.py

# Análisis de patrones
python utils/analysis.py --collection capec_patterns

# Backup de Milvus
python utils/milvus_backup.py export --collection capec_patterns

# Búsqueda de patrones
python search_patterns.py "cross-site scripting"
```

### Ejemplos

```bash
# Ejemplo de búsqueda
python examples/search_example.py "buffer overflow"

# Ejemplo de caché
python examples/cache_example.py

# Ejemplo de análisis
python examples/analysis_example.py
```

## 📋 Requisitos

- Python 3.8+
- Docker & Docker Compose
- 4GB RAM mínimo
- 10GB espacio en disco

### Dependencias Python

Ver [requirements.txt](requirements.txt) para la lista completa.

Principales:
- sentence-transformers
- pymilvus
- fastapi
- ollama (opcional, para LLM)

## 🔧 Configuración

Todas las configuraciones están centralizadas en `config.py`.

### Variables de Entorno

```bash
export MILVUS_HOST=localhost
export MILVUS_PORT=19530
export OLLAMA_HOST=http://localhost:11434
export API_PORT=8000
export ENABLE_CACHE=true
```

Ver [FEATURES.md](FEATURES.md#configuración-centralizada) para más detalles.

## 🧪 Tests

```bash
# Ejecutar todos los tests
python -m unittest discover tests -v

# Test específico
python -m unittest tests.test_config -v
```

## 📊 Análisis y Reportes

```bash
# Generar reporte completo
python utils/analysis.py --output report.txt

# Buscar patrones específicos
python utils/analysis.py --search "injection"

# Ver estadísticas
python -c "from utils.analysis import CAPECAnalyzer; \
  analyzer = CAPECAnalyzer('capec_patterns'); \
  print(analyzer.get_basic_stats())"
```

## 💾 Backup y Restauración

```bash
# Exportar colección
python utils/milvus_backup.py export --collection capec_patterns

# Importar colección
python utils/milvus_backup.py import --import-dir backups/capec_patterns_20231201
```

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Por favor:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/nueva-caracteristica`)
3. Añade tests si es necesario
4. Commit tus cambios (`git commit -am 'Añadir nueva característica'`)
5. Push a la rama (`git push origin feature/nueva-caracteristica`)
6. Crea un Pull Request

## 📄 Licencia

[Licencia MIT](LICENSE)

## 🙏 Agradecimientos

- [MITRE CAPEC](https://capec.mitre.org/) por la base de datos de patrones
- [Milvus](https://milvus.io/) por la base de datos vectorial
- [sentence-transformers](https://www.sbert.net/) por los modelos de embeddings
- [Ollama](https://ollama.ai/) por la integración con LLMs

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
