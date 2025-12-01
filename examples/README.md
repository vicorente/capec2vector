# Ejemplos de uso - capec2vector

Esta carpeta contiene ejemplos prácticos de cómo usar las características de capec2vector.

## 📋 Ejemplos disponibles

### 1. `search_example.py` - Búsqueda de patrones

Demuestra cómo usar la API REST para buscar patrones CAPEC.

**Uso:**

```bash
# Ejecutar ejemplos predefinidos
python examples/search_example.py

# Búsqueda personalizada
python examples/search_example.py "SQL injection in web applications"
```

**Requisitos:**
- API corriendo en http://localhost:8000
- Colección de Milvus cargada

**Aprenderás:**
- Cómo hacer búsquedas semánticas
- Interpretar scores de similitud
- Manejar respuestas de la API

---

### 2. `cache_example.py` - Sistema de caché

Demuestra el uso del sistema de caché para embeddings.

**Uso:**

```bash
python examples/cache_example.py
```

**Requisitos:**
- Solo Python (no requiere servicios externos)

**Aprenderás:**
- Guardar y recuperar embeddings del caché
- Ver estadísticas del caché
- Gestionar expiración (TTL)
- Limpiar el caché

---

### 3. `analysis_example.py` - Análisis de patrones

Demuestra las herramientas de análisis estadístico.

**Uso:**

```bash
python examples/analysis_example.py
```

**Requisitos:**
- Milvus corriendo
- Colección `capec_patterns` cargada

**Aprenderás:**
- Obtener estadísticas básicas
- Analizar distribución de severidad
- Analizar probabilidad de ataque
- Buscar patrones por palabra clave
- Generar reportes completos

---

## 🚀 Inicio rápido

Para ejecutar todos los ejemplos:

```bash
# 1. Asegúrate de que los servicios están corriendo
docker-compose up -d

# 2. Asegúrate de que la colección está cargada
python embeddings.py

# 3. Inicia la API (en otra terminal)
uvicorn ollama_milvus_bridge:app --reload

# 4. Ejecuta los ejemplos
python examples/search_example.py
python examples/cache_example.py
python examples/analysis_example.py
```

## 📝 Notas

- Todos los ejemplos incluyen manejo de errores y mensajes informativos
- Los ejemplos son interactivos (puedes presionar Enter para continuar)
- Se pueden usar como base para tus propios scripts
- Están documentados con comentarios explicativos

## 🎯 Casos de uso

### Para desarrolladores
- Aprende a integrar capec2vector en tus aplicaciones
- Entiende cómo funciona el sistema de caché
- Ve ejemplos de llamadas a la API

### Para analistas de seguridad
- Aprende a buscar patrones de ataque específicos
- Genera reportes estadísticos
- Analiza distribuciones de severidad y probabilidad

### Para administradores
- Monitorea el uso del caché
- Genera reportes periódicos
- Entiende el estado del sistema

## 💡 Tips

1. **Modifica los ejemplos**: Cambia los parámetros para ver diferentes resultados
2. **Combina ejemplos**: Usa ideas de múltiples ejemplos en tus scripts
3. **Lee los comentarios**: Cada ejemplo está bien documentado
4. **Experimenta**: Los ejemplos son seguros de ejecutar

## 🤝 Contribuir

Si creas un ejemplo útil, considera contribuirlo:

1. Añade un nuevo archivo `mi_ejemplo.py`
2. Documéntalo bien con comentarios
3. Añádelo a este README
4. Haz un Pull Request

## 📚 Recursos adicionales

- [QUICKSTART.md](../QUICKSTART.md) - Guía de inicio rápido
- [FEATURES.md](../FEATURES.md) - Documentación de características
- [API_DOCS.md](../API_DOCS.md) - Documentación de la API
- [README.md](../README.md) - Documentación principal
