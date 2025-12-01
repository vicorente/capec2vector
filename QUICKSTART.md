# 🚀 Guía de Inicio Rápido - capec2vector

Esta guía te ayudará a poner en marcha el proyecto capec2vector en minutos.

## 📋 Pre-requisitos

- Python 3.8 o superior
- Docker y Docker Compose
- 4GB de RAM disponible (mínimo)
- 10GB de espacio en disco

## ⚡ Instalación Rápida

### 1. Clonar el repositorio

```bash
git clone https://github.com/yourusername/capec2vector.git
cd capec2vector
```

### 2. Crear y activar entorno virtual

```bash
# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Verificar instalación

```bash
python setup.py
```

Deberías ver algo como:

```
============================================================
VERIFICACIÓN DE SETUP - capec2vector
============================================================

✓ Python 3.12.3
✓ Dependencias Python
✓ Directorios del proyecto
✓ Archivo CAPEC encontrado
✓ Docker

Resultado: 5/7 verificaciones pasadas
```

## 🐳 Iniciar servicios Docker

### 1. Levantar Milvus, etcd y MinIO

```bash
docker-compose up -d
```

### 2. Verificar que los servicios están corriendo

```bash
docker-compose ps
```

Deberías ver tres contenedores corriendo:
- `milvus-standalone`
- `milvus-etcd`
- `milvus-minio`

### 3. Esperar a que Milvus esté listo (30-60 segundos)

```bash
# Verificar logs
docker-compose logs milvus-standalone | tail -20
```

## 📊 Generar embeddings y cargar datos

### Ejecutar el script de embeddings

```bash
python embeddings.py
```

Esto hará:
1. ✅ Parsear el archivo XML de CAPEC
2. ✅ Limpiar y normalizar los datos
3. ✅ Generar embeddings vectoriales
4. ✅ Crear la colección en Milvus
5. ✅ Cargar todos los patrones CAPEC

**Tiempo estimado:** 5-15 minutos dependiendo de tu hardware.

**Salida esperada:**

```
2023-12-01 18:30:00 - INFO - Conectado a Milvus
2023-12-01 18:30:01 - INFO - Eliminando colección existente capec_patterns...
2023-12-01 18:30:02 - INFO - Colección capec_patterns creada
2023-12-01 18:30:05 - INFO - Cargando modelo de embeddings...
2023-12-01 18:35:20 - INFO - Procesando patrón 559/559
2023-12-01 18:35:25 - INFO - ✓ 559 patrones insertados en Milvus
```

## 🌐 Iniciar la API web

### Con uvicorn (recomendado)

```bash
uvicorn ollama_milvus_bridge:app --host 0.0.0.0 --port 8000 --reload
```

### Verificar que funciona

Abre tu navegador en: **http://localhost:8000**

Deberías ver la interfaz web del buscador de patrones CAPEC.

## 🧪 Probar la API

### 1. Verificar salud de la API

```bash
curl http://localhost:8000/health
```

### 2. Realizar una búsqueda

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "SQL injection", "top_k": 3}'
```

### 3. Usar el cliente Python

```python
import requests

# Buscar patrones
response = requests.post(
    "http://localhost:8000/search",
    json={"query": "cross-site scripting", "top_k": 5}
)

results = response.json()
for pattern in results["results"]:
    print(f"{pattern['pattern_id']}: {pattern['name']}")
```

## 🤖 Configurar Ollama (Opcional)

Si quieres usar la funcionalidad de LLM para respuestas contextualizadas:

### 1. Instalar Ollama

```bash
# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Mac
brew install ollama

# Windows
# Descargar desde https://ollama.ai/download
```

### 2. Descargar un modelo

```bash
# Modelo recomendado (3.8GB)
ollama pull qwen2.5-coder:7b

# O un modelo más ligero
ollama pull llama2:7b
```

### 3. Iniciar el servicio Ollama

```bash
ollama serve
```

### 4. Configurar la URL en config.py (si es necesario)

```python
# Si Ollama está en otro host
OLLAMA_HOST = "http://localhost:11434"
```

### 5. Probar con Ollama

```bash
curl -X POST http://localhost:8000/ollama/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are buffer overflow attacks?", "top_k": 3}'
```

## 📊 Herramientas adicionales

### Análisis de patrones

```bash
# Generar reporte de estadísticas
python utils/analysis.py --collection capec_patterns --output report.txt

# Buscar patrones por palabra clave
python utils/analysis.py --search "injection"
```

### Backup de Milvus

```bash
# Crear backup
python utils/milvus_backup.py export --collection capec_patterns --output-dir ./backups

# Restaurar backup
python utils/milvus_backup.py import --import-dir ./backups/capec_patterns_20231201_120000
```

### Verificar configuración

```bash
# Ver configuración actual
python config.py
```

### Ejecutar tests

```bash
# Todos los tests
python -m unittest discover tests -v

# Test específico
python -m unittest tests.test_config -v
```

## 🎯 Flujo de trabajo completo

Para uso diario, sigue este flujo:

```bash
# 1. Activar entorno virtual
source .venv/bin/activate

# 2. Verificar servicios Docker
docker-compose ps

# 3. Iniciar API (en una terminal)
uvicorn ollama_milvus_bridge:app --reload

# 4. Usar la interfaz web
# Abrir http://localhost:8000 en el navegador

# 5. O usar la API desde Python
python my_script.py
```

## 🔧 Solución de problemas comunes

### Error: "No module named 'pymilvus'"

```bash
pip install -r requirements.txt
```

### Error: "Connection refused" al conectar a Milvus

```bash
# Verificar que Docker está corriendo
docker-compose ps

# Reiniciar servicios
docker-compose restart

# Ver logs
docker-compose logs milvus-standalone
```

### Error: Puerto 8000 ocupado

```bash
# Matar proceso en puerto 8000
sudo lsof -ti:8000 | xargs kill -9

# O usar otro puerto
uvicorn ollama_milvus_bridge:app --port 8001
```

### La generación de embeddings es muy lenta

Esto es normal, especialmente en CPU. Considera:
- Usar un modelo más ligero (all-MiniLM-L6-v2)
- Activar GPU si está disponible
- Reducir el número de patrones procesados (para testing)

### Ollama no responde

```bash
# Verificar que Ollama está corriendo
curl http://localhost:11434/api/tags

# Reiniciar Ollama
ollama serve
```

## 📚 Siguientes pasos

Ahora que tienes el sistema funcionando:

1. 📖 Lee [FEATURES.md](FEATURES.md) para conocer todas las características
2. 🔌 Consulta [API_DOCS.md](API_DOCS.md) para detalles de la API
3. 🧪 Explora los ejemplos en `examples/` (si existen)
4. 🛠️ Personaliza `config.py` según tus necesidades
5. 📊 Genera reportes de análisis con `utils/analysis.py`
6. 💾 Configura backups regulares con `utils/milvus_backup.py`

## 🤝 Obtener ayuda

Si encuentras problemas:

1. Revisa los logs: `docker-compose logs`
2. Ejecuta `python setup.py` para diagnóstico
3. Verifica el endpoint `/health` de la API
4. Consulta los issues en GitHub
5. Lee la documentación completa en README.md

## 🎉 ¡Listo!

Ahora tienes un sistema completo de búsqueda semántica de patrones CAPEC con:

- ✅ Base de datos vectorial (Milvus)
- ✅ Embeddings de texto avanzados
- ✅ API REST completa
- ✅ Interfaz web interactiva
- ✅ Integración con LLM (opcional)
- ✅ Herramientas de análisis y backup

¡Disfruta explorando patrones de ataque CAPEC! 🔐
