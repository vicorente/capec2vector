from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
from ollama import Client, ChatResponse
import uvicorn
import requests
import json
import os
import socket
from typing import List
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuración
MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
COLLECTION_NAME = "capec_patterns"
OLLAMA_HOST = "http://172.16.5.180:11434"
HOST = "0.0.0.0"  # Permite conexiones desde cualquier IP

# Inicializar cliente de Ollama
logger.info(f"Inicializando cliente de Ollama en {OLLAMA_HOST}")
ollama_client = Client(host=OLLAMA_HOST)

class Pattern(BaseModel):
    pattern_id: str
    name: str
    description: str
    similarity_score: float

class SearchResponse(BaseModel):
    answer: str
    relevant_patterns: List[Pattern]

def find_available_port(start_port=8000, max_port=8999):
    """Encuentra un puerto disponible en el rango especificado"""
    for port in range(start_port, max_port + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((HOST, port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No se encontraron puertos disponibles en el rango {start_port}-{max_port}")

# Intentar usar el puerto especificado en la variable de entorno o encontrar uno disponible
try:
    PORT = int(os.getenv("PORT", "8000"))
    # Verificar si el puerto está disponible
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
except (OSError, ValueError):
    print(f"Puerto {PORT} no disponible, buscando uno alternativo...")
    PORT = find_available_port()

app = FastAPI(title="CAPEC Search API")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas las origenes (ajusta según tus necesidades de seguridad)
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos
    allow_headers=["*"],  # Permite todos los headers
)

# Configurar templates
templates = Jinja2Templates(directory="templates")

# Configurar archivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")

class SearchQuery(BaseModel):
    query: str
    top_k: int = 5

class OllamaQuery(BaseModel):
    query: str

def connect_to_milvus():
    """Establece conexión con Milvus"""
    try:
        logger.info(f"Intentando conectar a Milvus en {MILVUS_HOST}:{MILVUS_PORT}")
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        collection = Collection(COLLECTION_NAME)
        collection.load()
        logger.info("Conexión a Milvus establecida exitosamente")
        return collection
    except Exception as e:
        logger.error(f"Error al conectar con Milvus: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al conectar con Milvus: {str(e)}")

def search_patterns(query_text: str, top_k: int = 5):
    """Realiza búsqueda semántica en Milvus"""
    try:
        logger.info(f"Iniciando búsqueda de patrones para: '{query_text}'")
        logger.info("Cargando modelo de embeddings...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Generando embedding para la consulta...")
        query_embedding = model.encode(query_text)
        
        collection = connect_to_milvus()
        
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }
        
        logger.info(f"Realizando búsqueda en Milvus con top_k={top_k}")
        results = collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["pattern_id", "name", "description"]
        )
        
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "pattern_id": hit.entity.get("pattern_id"),
                    "name": hit.entity.get("name"),
                    "description": hit.entity.get("description"),
                    "similarity_score": 1 - hit.score
                })
        
        logger.info(f"Búsqueda completada. Encontrados {len(formatted_results)} patrones")
        return formatted_results
    
    except Exception as e:
        logger.error(f"Error durante la búsqueda: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error durante la búsqueda: {str(e)}")
    finally:
        logger.info("Cerrando conexión con Milvus")
        connections.disconnect("default")

@app.post("/search")
async def search_capec_patterns(search_query: SearchQuery):
    """Endpoint para búsqueda de patrones CAPEC"""
    try:
        results = search_patterns(search_query.query, search_query.top_k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def create_ollama_prompt(results):
    """Crea un prompt para Ollama con los resultados de la búsqueda"""
    logger.info("Creando prompt para Ollama")
    prompt = "Basado en los siguientes patrones de ataque CAPEC, por favor responde a la pregunta del usuario:\n\n"
    
    for i, result in enumerate(results, 1):
        prompt += f"{i}. {result['name']} (ID: {result['pattern_id']})\n"
        prompt += f"Descripción: {result['description']}\n"
        prompt += f"Relevancia: {result['similarity_score']:.4f}\n\n"
    
    prompt += f"""Para cada patrón de ataque, proporciona una breve descripción y el ID del patrón. Además, 
    proporciona una explicación detallada del patrón de ataque y cómo se puede explotar.	
    Proporciona un ejemplo de código en python para cada patrón de ataque, o un ejemplo de comando en terminal si es aplicable.
    Si encuentras un patrón etiquetado como "DEPRECATED", "DUPLICATED", "OBSOLETE" o "DRAFT", busca el patrón que lo reemplaza y proporciona la información de ese patrón.
    No te saltes ningún patrón, responde a todas las preguntas. Si hay un patrón sobre el que no tienes información, presenta una respuesta informativa.
    """
    logger.info("Prompt creado exitosamente")
    return prompt

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Sirve la página principal"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ollama/query", response_model=SearchResponse)
async def ollama_query(query_data: OllamaQuery):
    """Endpoint que integra la búsqueda en Milvus con Ollama"""
    try:
        logger.info(f"Recibida nueva consulta: '{query_data.query}'")
        
        # Primero, buscar patrones relevantes
        logger.info("Iniciando búsqueda de patrones en Milvus")
        results = search_patterns(query_data.query, top_k=3)
        
        # Crear prompt para Ollama
        logger.info("Preparando prompt para Ollama")
        prompt = create_ollama_prompt(results)
        logger.info(f"Prompt para Ollama: '{prompt}'")
        # Realizar consulta a Ollama usando el cliente
        response: ChatResponse = ollama_client.chat(
            #model="deepseek-r1:70b",
            model="qwen2.5:72b",
            messages=[
                {
                    "role": "system",
                    "content":  f"""
                    Eres un experto en ciberseguridad que analiza patrones de ataque CAPEC y proporciona respuestas detalladas y útiles.
                    Elabora las respuestas en formato markdown, con resaltado de código, listas, tablas, y aplicando saltos de linea para mejorar la legibilidad.
                    """
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            format=SearchResponse.model_json_schema(),
            options={"temperature": 0.7}
        )
        
        # Procesar la respuesta
        logger.info("Procesando respuesta de Ollama")
        try:
            # Intentar parsear la respuesta como JSON
            response_data = json.loads(response["message"]["content"])
            answer = response_data.get("answer", "")
        except json.JSONDecodeError:
            # Si no es JSON válido, usar el texto directamente
            logger.warning("La respuesta no es JSON válido, usando texto directo")
            answer = response["message"]["content"]
        
        logger.info("Consulta completada exitosamente")
        return {
            "answer": answer,
            "relevant_patterns": results
        }
        
    except Exception as e:
        logger.error(f"Error en el endpoint /ollama/query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("Iniciando servidor CAPEC Search API")
    logger.info(f"Host: {HOST}")
    logger.info(f"Puerto: {PORT}")
    logger.info(f"URL de acceso: http://{HOST}:{PORT}")
    logger.info(f"URL remota: http://<IP_DEL_SERVIDOR>:{PORT}")
    logger.info("=" * 50)
    
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        proxy_headers=True,
        forwarded_allow_ips="*",
        log_level="info"
    ) 