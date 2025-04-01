from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
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
from typing import List, Dict
import logging
from datetime import datetime
import asyncio
from collections import defaultdict

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
    top_k: int = 10 # Número de resultados a devolver. Para devolver ilimitados usar 0

class OllamaQuery(BaseModel):
    query: str

# Almacenamiento para los controladores de streaming activos
active_streams: Dict[str, asyncio.Task] = {}

# Variable global para el modelo
model = None

def get_embedding_model():
    """Singleton para el modelo de embeddings"""
    global model
    if model is None:
        logger.info("Inicializando modelo de embeddings...")
        model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
    return model

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

        # Usar el singleton del modelo
        model = get_embedding_model()

        # Generar embedding solo para la consulta
        logger.info("Generando embedding para la consulta...")
        query_embedding = model.encode(query_text)

        # Conectar a Milvus
        collection = connect_to_milvus()

        # Parámetros de búsqueda optimizados
        search_params = {
            "metric_type": "L2",
            "params": {
                "nprobe": 10,  # Número de clusters a buscar
                "ef": 64      # Factor de exploración
            }
        }

        # Realizar búsqueda vectorial
        logger.info(f"Realizando búsqueda en Milvus con top_k={top_k}")
        results = collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k * 2,  # Duplicar límite para compensar patrones deprecados
            output_fields=[
                "pattern_id",
                "name",
                "description",
                "status",
                "Typical_Severity",
                "Likelihood_Of_Attack",
                "Prerequisites",
                "Resources_Required",
                "Mitigations",
                "Example_Instances",
            ],
        )

        # Procesar resultados
        formatted_results = []
        deprecated_patterns = []

        for hits in results:
            for hit in hits:
                pattern = {
                    "pattern_id": hit.entity.get("pattern_id"),
                    "name": hit.entity.get("name"),
                    "description": hit.entity.get("description"),
                    "similarity_score": float(1 - hit.score),  # Convertir a float para serialización
                    "status": hit.entity.get("status"),
                    "typical_severity": hit.entity.get("Typical_Severity"),
                    "likelihood": hit.entity.get("Likelihood_Of_Attack"),
                    "prerequisites": hit.entity.get("Prerequisites"),
                    "resources_required": hit.entity.get("Resources_Required"),
                    "mitigations": hit.entity.get("Mitigations"),
                    "examples": hit.entity.get("Example_Instances"),
                }

                # Validar el estado del patrón
                if pattern["status"] and pattern["status"].lower() == "deprecated":
                    logger.info(f"Patrón deprecado encontrado: {pattern['pattern_id']}")
                    deprecated_patterns.append(pattern)
                else:
                    logger.info(f"Patrón encontrado: {pattern['pattern_id']}")
                    formatted_results.append(pattern)

        # Si tenemos menos resultados que top_k, buscar reemplazos
        if len(formatted_results) < top_k:
            for deprecated_pattern in deprecated_patterns:
                replacements = find_replacement_patterns(deprecated_pattern["description"])
                for replacement in replacements:
                    if len(formatted_results) >= top_k:
                        break
                    if not any(r["pattern_id"] == replacement["pattern_id"] for r in formatted_results):
                        formatted_results.append(replacement)
                        logger.info(f"Agregado patrón de reemplazo: {replacement['pattern_id']}")

        return formatted_results[:top_k]

    except Exception as e:
        logger.error(f"Error durante la búsqueda: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        logger.info("Cerrando conexión con Milvus")
        connections.disconnect("default")

def find_replacement_patterns(description: str) -> List[dict]:
    """Busca patrones de reemplazo en la descripción de un patrón deprecado"""
    try:
        # Buscar IDs de patrones en el formato CAPEC-XXX
        import re
        replacement_ids = re.findall(r'CAPEC-(\d+)', description)
        
        if not replacement_ids:
            return []
        
        collection = connect_to_milvus()
        
        # Buscar los patrones de reemplazo
        replacement_patterns = []
        for pattern_id in replacement_ids:
            try:
                results = collection.query(
                    expr=f'pattern_id == "CAPEC-{pattern_id}"',
                    output_fields=["pattern_id", "name", "description"]
                )
                if results:
                    pattern = results[0]
                    replacement_patterns.append({
                        "pattern_id": pattern.get("pattern_id"),
                        "name": pattern.get("name"),
                        "description": pattern.get("description"),
                        "similarity_score": 1.0,  # Máxima relevancia para patrones de reemplazo
                        "status": "REPLACEMENT"
                    })
            except Exception as e:
                logger.warning(f"Error al buscar patrón de reemplazo CAPEC-{pattern_id}: {str(e)}")
        
        return replacement_patterns
    
    except Exception as e:
        logger.error(f"Error al buscar patrones de reemplazo: {str(e)}")
        return []
    finally:
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
    De todos los patrones, responde a todas las preguntas. Si hay un patrón sobre el que no tienes información, presenta una respuesta informativa.
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
        results = search_patterns(query_data.query, top_k=10)
        
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

@app.post("/ollama/stop/{stream_id}")
async def stop_generation(stream_id: str):
    """Endpoint para detener la generación de un stream específico"""
    try:
        if stream_id in active_streams:
            # Cancelar la tarea de streaming
            active_streams[stream_id].cancel()
            del active_streams[stream_id]
            return {"status": "success", "message": "Generación detenida exitosamente"}
        return {"status": "error", "message": "Stream no encontrado"}
    except Exception as e:
        logger.error(f"Error al detener la generación: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ollama/query/stream")
async def ollama_query_stream(query_data: OllamaQuery):
    """Endpoint que integra la búsqueda en Milvus con Ollama usando streaming"""
    try:
        logger.info(f"Recibida nueva consulta streaming: '{query_data.query}'")

        # Generar un ID único para este stream
        stream_id = f"stream_{datetime.now().timestamp()}"

        # Primero, buscar patrones relevantes
        logger.info("Iniciando búsqueda de patrones en Milvus")
        results = search_patterns(query_data.query, top_k=10)

        # Crear y almacenar la tarea de streaming
        async def generate():
            try:
                # Enviar los patrones relevantes inmediatamente
                # logger.info(f"resultados encontrados: {results}")
                simplified_results = [
                    {
                        "pattern_id": pattern["pattern_id"],
                        "name": pattern["name"],                       
                    }
                    for pattern in results
                ]
                yield f"data: {json.dumps({'patterns': simplified_results, 'stream_id': stream_id}, ensure_ascii=False)}\n\n"

                # Crear prompt para Ollama
                # logger.info("Preparando prompt para Ollama")
                # prompt = create_ollama_prompt(results)

                # # Realizar consulta a Ollama usando el cliente con streaming
                # response = ollama_client.chat(
                #     model="qwen2.5:72b",
                #     messages=[
                #         {
                #             "role": "system",
                #             "content": f"""
                #             Eres un experto en ciberseguridad que analiza patrones de ataque CAPEC y proporciona respuestas detalladas y útiles.
                #             Elabora las respuestas en formato markdown, con resaltado de código, listas, tablas, y aplicando saltos de linea para mejorar la legibilidad.
                #             """
                #         },
                #         {
                #             "role": "user",
                #             "content": prompt
                #         }
                #     ],
                #     stream=True,
                #     options={"temperature": 0.7}
                # )

                # # Procesar la respuesta en streaming
                # for chunk in response:
                #     if chunk and "message" in chunk and "content" in chunk["message"]:
                #         # Escapar caracteres especiales y asegurar que el JSON sea válido
                #         content = chunk["message"]["content"]
                #         try:
                #             # Intentar codificar y decodificar para asegurar que el contenido es válido
                #             content = content.encode('utf-8').decode('utf-8')
                #             data = {"chunk": content}
                #             yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                #         except Exception as e:
                #             logger.error(f"Error al procesar chunk: {str(e)}")
                #             continue

                yield "data:\n\n"

            except asyncio.CancelledError:
                logger.info(f"Stream {stream_id} cancelado por el usuario")
                yield f"data: {json.dumps({'status': 'cancelled'}, ensure_ascii=False)}\n\n"
            except Exception as e:
                logger.error(f"Error en el streaming: {str(e)}")
                try:
                    yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
                except:
                    yield "data:\n\n"
            finally:
                if stream_id in active_streams:
                    del active_streams[stream_id]

        # No crear la tarea aquí, solo retornar el StreamingResponse directamente
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )

    except Exception as e:
        logger.error(f"Error en el endpoint /ollama/query/stream: {str(e)}")
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
