# Documentación de la API - capec2vector

Esta documentación describe todos los endpoints disponibles en la API web de capec2vector.

## 🚀 Inicio Rápido

### Iniciar el servidor

```bash
uvicorn ollama_milvus_bridge:app --host 0.0.0.0 --port 8000 --reload
```

### URL Base

```
http://localhost:8000
```

---

## 📑 Índice de Endpoints

1. [GET /](#get-) - Página principal (interfaz web)
2. [POST /search](#post-search) - Búsqueda semántica en Milvus
3. [POST /ollama/query](#post-ollamaquery) - Búsqueda con respuesta de LLM
4. [GET /patterns/all](#get-patternsall) - Listar todos los patrones
5. [GET /patterns/{pattern_id}](#get-patternspattern_id) - Obtener patrón específico
6. [GET /health](#get-health) - Estado de salud de la API

---

## Endpoints

### GET /

**Descripción:** Página principal con interfaz web interactiva para buscar patrones CAPEC.

**Respuesta:**
- Tipo: `text/html`
- Contenido: Página HTML con interfaz de usuario

**Ejemplo:**

```bash
curl http://localhost:8000/
```

---

### POST /search

**Descripción:** Realiza una búsqueda semántica de patrones CAPEC basada en una consulta en lenguaje natural.

**Request Body:**

```json
{
  "query": "SQL injection attacks",
  "top_k": 5
}
```

**Parámetros:**
- `query` (string, requerido): Texto de búsqueda en lenguaje natural
- `top_k` (integer, opcional): Número de resultados a retornar (default: 5, max: 100)

**Respuesta Exitosa (200):**

```json
{
  "query": "SQL injection attacks",
  "results": [
    {
      "pattern_id": "66",
      "name": "SQL Injection",
      "description": "An attacker crafts special user-controllable input...",
      "similarity_score": 0.85,
      "status": "Stable",
      "abstraction": "Standard",
      "typical_severity": "High",
      "likelihood_of_attack": "High"
    }
  ],
  "total_results": 5,
  "execution_time_ms": 145
}
```

**Errores:**
- `400 Bad Request`: Query vacío o top_k inválido
- `500 Internal Server Error`: Error de conexión con Milvus

**Ejemplo con curl:**

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "SQL injection", "top_k": 3}'
```

**Ejemplo con Python:**

```python
import requests

response = requests.post(
    "http://localhost:8000/search",
    json={"query": "SQL injection attacks", "top_k": 5}
)

results = response.json()
for result in results["results"]:
    print(f"{result['pattern_id']}: {result['name']}")
```

---

### POST /ollama/query

**Descripción:** Realiza una búsqueda semántica y genera una respuesta contextualizada usando un LLM (Ollama).

**Request Body:**

```json
{
  "query": "How can I protect against SQL injection?",
  "top_k": 5,
  "model": "qwen2.5-coder:7b"
}
```

**Parámetros:**
- `query` (string, requerido): Pregunta o consulta en lenguaje natural
- `top_k` (integer, opcional): Número de patrones relevantes a considerar (default: 5)
- `model` (string, opcional): Modelo de Ollama a usar (default: qwen2.5-coder:7b)

**Respuesta Exitosa (200):**

```json
{
  "answer": "Para proteger contra SQL injection, debes:\n1. Usar prepared statements...",
  "relevant_patterns": [
    {
      "pattern_id": "66",
      "name": "SQL Injection",
      "description": "...",
      "similarity_score": 0.85
    }
  ],
  "model_used": "qwen2.5-coder:7b",
  "query": "How can I protect against SQL injection?",
  "execution_time_ms": 2340
}
```

**Errores:**
- `400 Bad Request`: Query vacío
- `500 Internal Server Error`: Error de conexión con Milvus u Ollama
- `503 Service Unavailable`: Ollama no disponible

**Ejemplo con curl:**

```bash
curl -X POST http://localhost:8000/ollama/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are cross-site scripting attacks?",
    "top_k": 3
  }'
```

**Ejemplo con Python:**

```python
import requests

response = requests.post(
    "http://localhost:8000/ollama/query",
    json={
        "query": "How do buffer overflow attacks work?",
        "top_k": 5,
        "model": "llama2:7b"
    }
)

result = response.json()
print("Respuesta:", result["answer"])
print(f"Patrones consultados: {len(result['relevant_patterns'])}")
```

---

### GET /patterns/all

**Descripción:** Obtiene una lista de todos los patrones CAPEC disponibles en la base de datos.

**Query Parameters:**
- `limit` (integer, opcional): Número máximo de resultados (default: 100, max: 1000)
- `offset` (integer, opcional): Número de resultados a saltar (default: 0)
- `status` (string, opcional): Filtrar por status (e.g., "Stable", "Draft")
- `abstraction` (string, opcional): Filtrar por nivel de abstracción

**Respuesta Exitosa (200):**

```json
{
  "patterns": [
    {
      "pattern_id": "1",
      "name": "Accessing Functionality Not Properly Constrained by ACLs",
      "status": "Stable",
      "abstraction": "Standard"
    }
  ],
  "total": 559,
  "limit": 100,
  "offset": 0
}
```

**Ejemplo con curl:**

```bash
curl "http://localhost:8000/patterns/all?limit=10&status=Stable"
```

---

### GET /patterns/{pattern_id}

**Descripción:** Obtiene información detallada de un patrón CAPEC específico.

**Path Parameters:**
- `pattern_id` (string, requerido): ID del patrón CAPEC

**Respuesta Exitosa (200):**

```json
{
  "pattern_id": "66",
  "name": "SQL Injection",
  "description": "Detailed description...",
  "status": "Stable",
  "abstraction": "Standard",
  "typical_severity": "High",
  "likelihood_of_attack": "High",
  "prerequisites": "...",
  "skills_required": "...",
  "mitigations": "...",
  "example_instances": "...",
  "related_attack_patterns": ["..."],
  "related_weaknesses": ["..."]
}
```

**Errores:**
- `404 Not Found`: Patrón no encontrado

**Ejemplo con curl:**

```bash
curl http://localhost:8000/patterns/66
```

**Ejemplo con Python:**

```python
import requests

response = requests.get("http://localhost:8000/patterns/66")
pattern = response.json()

print(f"Nombre: {pattern['name']}")
print(f"Severidad: {pattern['typical_severity']}")
print(f"Mitigaciones: {pattern['mitigations']}")
```

---

### GET /health

**Descripción:** Verifica el estado de salud de la API y sus dependencias.

**Respuesta Exitosa (200):**

```json
{
  "status": "healthy",
  "timestamp": "2023-12-01T18:30:00.000Z",
  "services": {
    "milvus": {
      "status": "connected",
      "host": "localhost",
      "port": 19530,
      "collection": "capec_patterns",
      "entities": 559
    },
    "ollama": {
      "status": "available",
      "host": "http://172.16.11.224:11434",
      "models": ["qwen2.5-coder:7b", "llama2:7b"]
    }
  },
  "version": "1.0.0"
}
```

**Respuesta con Problemas (503):**

```json
{
  "status": "degraded",
  "timestamp": "2023-12-01T18:30:00.000Z",
  "services": {
    "milvus": {
      "status": "error",
      "error": "Connection refused"
    },
    "ollama": {
      "status": "error",
      "error": "Service not available"
    }
  }
}
```

**Ejemplo con curl:**

```bash
curl http://localhost:8000/health
```

---

## 🔐 Autenticación

Actualmente la API no requiere autenticación, pero se recomienda implementar autenticación para entornos de producción.

### Recomendaciones de seguridad:

1. **API Keys**: Implementar autenticación con API keys
2. **Rate Limiting**: Limitar número de requests por IP
3. **CORS**: Configurar CORS apropiadamente
4. **HTTPS**: Usar HTTPS en producción

---

## 📊 Códigos de Estado HTTP

| Código | Significado |
|--------|-------------|
| 200 | OK - Petición exitosa |
| 400 | Bad Request - Parámetros inválidos |
| 404 | Not Found - Recurso no encontrado |
| 500 | Internal Server Error - Error del servidor |
| 503 | Service Unavailable - Servicio no disponible |

---

## 🔍 Ejemplos de Uso Completos

### Ejemplo 1: Búsqueda básica de patrones

```python
import requests

def search_patterns(query, top_k=5):
    """Busca patrones CAPEC por consulta en lenguaje natural"""
    url = "http://localhost:8000/search"
    
    response = requests.post(url, json={
        "query": query,
        "top_k": top_k
    })
    
    if response.status_code == 200:
        results = response.json()
        print(f"Encontrados {results['total_results']} resultados:")
        
        for i, pattern in enumerate(results["results"], 1):
            print(f"\n{i}. {pattern['name']} (ID: {pattern['pattern_id']})")
            print(f"   Severidad: {pattern['typical_severity']}")
            print(f"   Similitud: {pattern['similarity_score']:.2f}")
    else:
        print(f"Error: {response.status_code}")

# Usar la función
search_patterns("injection attacks", top_k=3)
```

### Ejemplo 2: Consulta con LLM

```python
import requests

def ask_ollama(question):
    """Hace una pregunta usando búsqueda semántica + LLM"""
    url = "http://localhost:8000/ollama/query"
    
    response = requests.post(url, json={
        "query": question,
        "top_k": 5
    })
    
    if response.status_code == 200:
        result = response.json()
        
        print("Respuesta del LLM:")
        print("-" * 80)
        print(result["answer"])
        print("-" * 80)
        
        print(f"\nPatrones consultados: {len(result['relevant_patterns'])}")
        for pattern in result["relevant_patterns"]:
            print(f"  - {pattern['name']}")
    else:
        print(f"Error: {response.status_code}")

# Usar la función
ask_ollama("What are the most dangerous web application attacks?")
```

### Ejemplo 3: Análisis de patrones por severidad

```python
import requests
from collections import Counter

def analyze_severity():
    """Analiza la distribución de severidades"""
    url = "http://localhost:8000/patterns/all"
    
    response = requests.get(url, params={"limit": 1000})
    
    if response.status_code == 200:
        data = response.json()
        patterns = data["patterns"]
        
        # Contar severidades
        severities = [p.get("typical_severity", "Unknown") 
                     for p in patterns]
        severity_counts = Counter(severities)
        
        print("Distribución de Severidad:")
        for severity, count in severity_counts.most_common():
            percentage = (count / len(patterns)) * 100
            print(f"  {severity}: {count} ({percentage:.1f}%)")

# Usar la función
analyze_severity()
```

---

## 🧪 Testing

### Test de conectividad

```bash
# Verificar que la API está corriendo
curl http://localhost:8000/health

# Test de búsqueda simple
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "top_k": 1}'
```

### Tests automatizados

```python
import requests
import unittest

class TestAPI(unittest.TestCase):
    BASE_URL = "http://localhost:8000"
    
    def test_health(self):
        response = requests.get(f"{self.BASE_URL}/health")
        self.assertEqual(response.status_code, 200)
    
    def test_search(self):
        response = requests.post(
            f"{self.BASE_URL}/search",
            json={"query": "SQL injection", "top_k": 3}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("results", data)
        self.assertLessEqual(len(data["results"]), 3)

if __name__ == "__main__":
    unittest.main()
```

---

## 📝 Notas

- Todos los endpoints retornan JSON excepto el endpoint raíz que retorna HTML
- Los tiempos de respuesta incluyen la generación de embeddings y búsqueda en Milvus
- Las consultas a Ollama pueden tardar varios segundos dependiendo del modelo
- Se recomienda usar `top_k` entre 3 y 10 para mejor balance entre relevancia y velocidad

---

## 🔗 Enlaces Útiles

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Milvus Documentation](https://milvus.io/docs)
- [Ollama Documentation](https://ollama.ai/docs)
- [CAPEC Database](https://capec.mitre.org/)
