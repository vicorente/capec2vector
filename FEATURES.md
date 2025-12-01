# Nuevas Características - capec2vector

Este documento describe las nuevas características añadidas al proyecto capec2vector.

## 📋 Tabla de Contenidos

1. [Configuración Centralizada](#configuración-centralizada)
2. [Sistema de Tests](#sistema-de-tests)
3. [Sistema de Caché](#sistema-de-caché)
4. [Validadores y Sanitización](#validadores-y-sanitización)
5. [Script de Inicialización](#script-de-inicialización)
6. [Backup y Restauración de Milvus](#backup-y-restauración-de-milvus)
7. [Herramientas de Análisis](#herramientas-de-análisis)

---

## Configuración Centralizada

### Archivo: `config.py`

Todas las configuraciones del proyecto están ahora centralizadas en un único archivo que:

- ✅ Soporta variables de entorno
- ✅ Proporciona valores por defecto sensatos
- ✅ Es fácil de mantener y modificar

**Uso:**

```python
from config import MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME

# Obtener resumen de configuración
from config import get_config_summary
print(get_config_summary())
```

**Variables de entorno soportadas:**

```bash
export MILVUS_HOST=localhost
export MILVUS_PORT=19530
export COLLECTION_NAME=capec_patterns
export EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1
export OLLAMA_HOST=http://localhost:11434
export API_PORT=8000
export ENABLE_CACHE=true
export LOG_LEVEL=INFO
```

---

## Sistema de Tests

### Directorio: `tests/`

Se han añadido tests unitarios para verificar la funcionalidad del sistema:

- `tests/test_config.py` - Tests de configuración

**Ejecutar tests:**

```bash
# Con unittest (no requiere pytest)
python -m unittest discover tests -v

# Con pytest (si está instalado)
pytest tests/ -v
```

**Cobertura actual:**
- ✅ Configuración de Milvus
- ✅ Configuración de embeddings
- ✅ Configuración de API
- ✅ Configuración de caché
- ✅ Validación de archivos del proyecto

---

## Sistema de Caché

### Archivo: `utils/cache.py`

Sistema de caché para embeddings que evita regenerar vectores ya calculados:

**Características:**
- 🔄 Cache automático de embeddings
- ⏰ TTL configurable (tiempo de expiración)
- 📊 Estadísticas de uso
- 🗑️ Limpieza automática de elementos expirados

**Uso:**

```python
from utils.cache import EmbeddingCache
from pathlib import Path

# Crear instancia de caché
cache = EmbeddingCache(Path(".cache"), ttl_seconds=3600)

# Guardar embedding
cache.set("mi texto", embedding_array)

# Recuperar embedding
embedding = cache.get("mi texto")

# Obtener estadísticas
stats = cache.get_stats()
print(f"Elementos en caché: {stats['total_items']}")
print(f"Tamaño total: {stats['total_size_mb']} MB")

# Limpiar caché
cache.clear()
```

---

## Validadores y Sanitización

### Archivo: `utils/validators.py`

Utilidades para validar y sanitizar datos antes de almacenarlos en Milvus:

**Funciones disponibles:**

```python
from utils.validators import (
    sanitize_text,
    validate_pattern_data,
    validate_embedding,
    truncate_field
)

# Sanitizar texto
clean_text = sanitize_text(raw_text, max_length=1000)

# Validar datos de patrón
is_valid = validate_pattern_data({
    "pattern_id": "123",
    "name": "SQL Injection",
    "description": "..."
})

# Validar embedding
is_valid_emb = validate_embedding(embedding_array)

# Truncar campos largos
truncated = truncate_field(long_text, max_length=5000, field_name="description")
```

**Beneficios:**
- 🛡️ Protección contra datos inválidos
- 🧹 Limpieza automática de caracteres problemáticos
- ✂️ Truncado seguro de campos largos
- 📝 Logging de problemas detectados

---

## Script de Inicialización

### Archivo: `setup.py`

Script interactivo para verificar y configurar el entorno del proyecto:

**Ejecución:**

```bash
python setup.py
```

**Verificaciones realizadas:**

1. ✅ Versión de Python (>= 3.8)
2. ✅ Dependencias Python instaladas
3. ✅ Directorios del proyecto creados
4. ✅ Datos CAPEC disponibles
5. ✅ Docker instalado y funcionando
6. ⚠️ Milvus corriendo (opcional)
7. ⚠️ Ollama disponible (opcional)

**Salida de ejemplo:**

```
============================================================
VERIFICACIÓN DE SETUP - capec2vector
============================================================

✓ Python 3.12.3
✓ Dependencias Python
✓ Directorios del proyecto
✓ Archivo CAPEC encontrado: capec_latest/capec_v3.9.xml
  Tamaño: 3.67 MB
✓ Docker

Resultado: 5/7 verificaciones pasadas
```

---

## Backup y Restauración de Milvus

### Archivo: `utils/milvus_backup.py`

Herramienta para realizar backups de colecciones de Milvus:

**Uso desde línea de comandos:**

```bash
# Listar colecciones
python utils/milvus_backup.py list

# Exportar colección
python utils/milvus_backup.py export --collection capec_patterns --output-dir ./backups

# Importar colección
python utils/milvus_backup.py import --import-dir ./backups/capec_patterns_20231201_120000

# Con host/puerto personalizados
python utils/milvus_backup.py export --collection capec_patterns --host 192.168.1.100 --port 19530
```

**Uso programático:**

```python
from utils.milvus_backup import MilvusBackup
from pathlib import Path

backup = MilvusBackup(host="localhost", port=19530)

# Exportar colección
export_path = backup.export_collection(
    collection_name="capec_patterns",
    output_dir=Path("./backups")
)

# Importar colección
backup.import_collection(
    import_dir=export_path,
    collection_name="capec_patterns_restored",
    overwrite=False
)

# Listar colecciones
collections = backup.list_collections()
```

**Formato de exportación:**
- `schema.json` - Esquema de la colección
- `metadata.json` - Metadata del backup
- `batch_*.pkl` - Datos en lotes (pickle)

---

## Herramientas de Análisis

### Archivo: `utils/analysis.py`

Herramientas para analizar y generar estadísticas sobre patrones CAPEC:

**Uso desde línea de comandos:**

```bash
# Generar reporte completo
python utils/analysis.py --collection capec_patterns

# Guardar reporte en archivo
python utils/analysis.py --collection capec_patterns --output reporte.txt

# Buscar patrones por palabra clave
python utils/analysis.py --collection capec_patterns --search "SQL injection"
```

**Uso programático:**

```python
from utils.analysis import CAPECAnalyzer

analyzer = CAPECAnalyzer("capec_patterns")

# Estadísticas básicas
stats = analyzer.get_basic_stats()
print(f"Total patrones: {stats['total_patterns']}")

# Distribución de severidad
severity = analyzer.get_severity_distribution()

# Distribución de probabilidad de ataque
likelihood = analyzer.get_likelihood_distribution()

# Buscar por palabra clave
results = analyzer.search_patterns_by_keyword("injection")

# Generar reporte completo
report = analyzer.generate_report(output_file="analysis_report.txt")
print(report)
```

**Métricas incluidas:**
- 📊 Total de patrones
- 🏷️ Distribución por status
- 🔍 Distribución por nivel de abstracción
- ⚠️ Distribución de severidad
- 🎯 Distribución de probabilidad de ataque
- 🔎 Búsqueda por palabras clave

**Ejemplo de reporte:**

```
================================================================================
REPORTE DE ANÁLISIS CAPEC
================================================================================

ESTADÍSTICAS BÁSICAS
--------------------------------------------------------------------------------
Total de patrones: 559

Distribución por Status:
  Stable: 450 (80.5%)
  Draft: 109 (19.5%)

Distribución por Abstraction:
  Detailed: 320 (57.2%)
  Standard: 189 (33.8%)
  Meta: 50 (8.9%)

DISTRIBUCIÓN DE SEVERIDAD
--------------------------------------------------------------------------------
High: 234
Medium: 201
Low: 98
Very High: 26

DISTRIBUCIÓN DE PROBABILIDAD DE ATAQUE
--------------------------------------------------------------------------------
Medium: 245
High: 189
Low: 125
```

---

## 🎯 Beneficios Generales

Las nuevas características proporcionan:

1. **Mejor Mantenibilidad**: Configuración centralizada y código más organizado
2. **Mayor Confiabilidad**: Tests automatizados y validación de datos
3. **Mejor Performance**: Sistema de caché para embeddings
4. **Facilidad de Uso**: Scripts de setup y análisis automáticos
5. **Recuperación ante Desastres**: Sistema de backup y restauración
6. **Insights**: Herramientas de análisis y generación de reportes

---

## 📚 Próximos Pasos

Para aprovechar estas características:

1. Ejecuta `python setup.py` para verificar tu entorno
2. Revisa `config.py` y ajusta según tus necesidades
3. Ejecuta los tests con `python -m unittest discover tests`
4. Considera usar el sistema de caché en tus flujos de trabajo
5. Realiza backups regulares de tus colecciones de Milvus
6. Genera reportes de análisis para entender tus datos

---

## 🤝 Contribuciones

Para añadir más funcionalidades:

1. Añade tests en el directorio `tests/`
2. Documenta las nuevas características en este archivo
3. Actualiza el README.md principal si es necesario
4. Asegúrate de que el código pase los tests existentes

---

## 📝 Notas

- Todas las utilidades soportan logging configurable
- Los directorios temporales (`.cache`, `logs`, `backups`) están en `.gitignore`
- El sistema de caché usa SHA256 para identificar textos únicos
- Los backups de Milvus se guardan en formato pickle para preservar tipos de datos
