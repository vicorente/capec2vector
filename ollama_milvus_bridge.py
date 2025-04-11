from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request, Query
from pydantic import BaseModel
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
from ollama import Client, ChatResponse
import uvicorn
import requests
import json
import os
import socket
from typing import List, Dict, Any
import logging
from datetime import datetime
import asyncio
from collections import defaultdict

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Configuración
MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
COLLECTION_NAME = "capec_patterns"
OLLAMA_HOST = "http://172.16.11.224:11434"
HOST = "0.0.0.0"  # Permite conexiones desde cualquier IP
KALI_API_PORT = int(os.environ.get("API_PORT", 5000))
KALI_API_BASE_URL = os.environ.get(
    "KALI_API_BASE_URL", f"http://172.16.11.111:{KALI_API_PORT}"
)
DEFAULT_MODEL = "qwen2.5-coder:7b"  # Default Ollama model
OLLAMA_PATTERN_RESPONSE = None  # Global variable to store Ollama's response

# Available Kali Linux tools mapped to their API endpoints
KALI_TOOLS = {
    "nmap": "/api/tools/nmap",
    "gobuster": "/api/tools/gobuster",
    "dirb": "/api/tools/dirb",
    "nikto": "/api/tools/nikto",
    "sqlmap": "/api/tools/sqlmap",
    "metasploit": "/api/tools/metasploit",
    "hydra": "/api/tools/hydra",
    "john": "/api/tools/john",
    "wpscan": "/api/tools/wpscan",
    "enum4linux": "/api/tools/enum4linux",
}

# Inicializar cliente de Ollama
logger.info(f"Inicializando cliente de Ollama en {OLLAMA_HOST}")
ollama_client = Client(host=OLLAMA_HOST)
conversation = []


class Pattern(BaseModel):
    pattern_id: str
    name: str
    description: str
    similarity_score: float


class SearchResponse(BaseModel):
    answer: str
    relevant_patterns: List[Pattern]


class OllamaPrompt(BaseModel):
    prompt: str
    pattern_id: str  # Adding pattern_id as a required field


# Intentar usar siempre el puerto 8000
# sudo lsof -i :8000 | grep LISTEN
PORT = 8000
try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
except OSError:
    logger.error(
        f"Puerto {PORT} no disponible. Por favor, libere el puerto y vuelva a intentar."
    )
    exit(1)

app = FastAPI(title="CAPEC Search API")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],  # Permite todas las origenes (ajusta según tus necesidades de seguridad)
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
    top_k: int = 10  # Número de resultados a devolver. Para devolver ilimitados usar 0


class OllamaQuery(BaseModel):
    query: str
    top_k: int = 10


# Almacenamiento para los controladores de streaming activos
active_streams: Dict[str, asyncio.Task] = {}

# Variable global para el modelo
model = None


def get_embedding_model():
    """Singleton para el modelo de embeddings"""
    global model
    if model is None:
        logger.info("Inicializando modelo de embeddings...")
        model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1", trust_remote_code=True
        )
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
        raise HTTPException(
            status_code=500, detail=f"Error al conectar con Milvus: {str(e)}"
        )


def search_patterns(query_text: str, top_k: int):
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
                "ef": 64,  # Factor de exploración
            },
        }

        # Realizar búsqueda vectorial
        logger.info(f"Realizando búsqueda en Milvus con top_k={top_k}")
        results = collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k * 2,  # Duplicar límite para compensar patrones deprecados
            output_fields=["pattern_id", "name", "description", "status"],
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
                    "similarity_score": float(
                        1 - hit.score
                    ),  # Convertir a float para serialización
                    "status": hit.entity.get("status"),
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
                replacements = find_replacement_patterns(
                    deprecated_pattern["description"]
                )
                for replacement in replacements:
                    if len(formatted_results) >= top_k:
                        break
                    if not any(
                        r["pattern_id"] == replacement["pattern_id"]
                        for r in formatted_results
                    ):
                        formatted_results.append(replacement)
                        logger.info(
                            f"Agregado patrón de reemplazo: {replacement['pattern_id']}"
                        )

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

        replacement_ids = re.findall(r"CAPEC-(\d+)", description)

        if not replacement_ids:
            return []

        collection = connect_to_milvus()

        # Buscar los patrones de reemplazo
        replacement_patterns = []
        for pattern_id in replacement_ids:
            try:
                results = collection.query(
                    expr=f'pattern_id == "CAPEC-{pattern_id}"',
                    output_fields=["pattern_id", "name", "description"],
                )
                if results:
                    pattern = results[0]
                    replacement_patterns.append(
                        {
                            "pattern_id": pattern.get("pattern_id"),
                            "name": pattern.get("name"),
                            "description": pattern.get("description"),
                            "similarity_score": 1.0,  # Máxima relevancia para patrones de reemplazo
                            "status": "REPLACEMENT",
                        }
                    )
            except Exception as e:
                logger.warning(
                    f"Error al buscar patrón de reemplazo CAPEC-{pattern_id}: {str(e)}"
                )

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
        results = search_patterns(query_data.query, query_data.top_k)

        # Crear prompt para Ollama
        logger.info("Preparando prompt para Ollama")
        prompt = create_ollama_prompt(results)
        logger.info(f"Prompt para Ollama: '{prompt}'")
        # Realizar consulta a Ollama usando el cliente
        response: ChatResponse = ollama_client.chat(
            # model="deepseek-r1:70b",
            model="qwen2.5-coder:7b",
            messages=[
                {
                    "role": "system",
                    "content": f"""
                    Eres un experto en ciberseguridad que analiza patrones de ataque CAPEC y proporciona respuestas detalladas y útiles.
                    Elabora las respuestas en formato markdown, con resaltado de código, listas, tablas, y aplicando saltos de linea para mejorar la legibilidad.
                    """,
                },
                {"role": "user", "content": prompt},
            ],
            format=SearchResponse.model_json_schema(),
            options={"temperature": 0.7},
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
        return {"answer": answer, "relevant_patterns": results}

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


@app.post("/patterns/query/stream")
async def patterns_query_stream(query_data: SearchQuery):
    try:
        logger.info(f"Recibida nueva consulta streaming: '{query_data.query}'")
        stream_id = f"stream_{datetime.now().timestamp()}"
        logger.info("Iniciando búsqueda de patrones en Milvus")
        results = search_patterns(query_data.query, query_data.top_k)

        async def generate():
            try:
                # Enviar los patrones relevantes con todos sus detalles
                patterns_with_details = [
                    {
                        "pattern_id": pattern["pattern_id"],
                        "name": pattern["name"],
                        "description": pattern["description"],
                    }
                    for pattern in results
                ]
                yield f"data: {json.dumps({'patterns': patterns_with_details, 'stream_id': stream_id}, ensure_ascii=False)}\n"

            except asyncio.CancelledError:
                logger.info(f"Stream {stream_id} cancelado por el usuario")
                yield f"data: {json.dumps({'status': 'cancelled'}, ensure_ascii=False)}\n"
            except Exception as e:
                logger.error(f"Error en el streaming: {str(e)}")
                try:
                    yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n"
                except:
                    yield "data:\n"
            finally:
                if stream_id in active_streams:
                    del active_streams[stream_id]

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    except Exception as e:
        logger.error(f"Error en el endpoint /ollama/query/stream: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ollama/analyze")
async def analyze_pattern(prompt_data: OllamaPrompt):
    """Endpoint para analizar un patrón específico usando Ollama"""
    try:
        logger.info("Recibida solicitud de análisis de patrón")
        prompt = prompt_data.prompt
        pattern_id = prompt_data.pattern_id

        if not prompt or not pattern_id:
            raise HTTPException(
                status_code=400, detail="Se requiere prompt y pattern_id"
            )

        # Buscar el patrón en Milvus
        try:
            collection = connect_to_milvus()
            pattern_details = collection.query(
                expr=f'pattern_id == "{pattern_id}"',
                output_fields=[
                    "pattern_id",
                    "name",
                    "description",
                    "status",
                    "abstraction",
                    "summary",
                    "alternate_terms",
                    "submission_date",
                    "submission_name",
                    "submission_organization",
                    "typical_severity",
                    "likelihood_of_attack",
                    "prerequisites",
                    "skills_required",
                    "resources_required",
                    "indicators",
                    "consequences",
                    "mitigations",
                    "example_instances",
                    "notes",
                    "related_attack_patterns",
                    "related_weaknesses",
                    "taxonomy_mappings",
                    "execution_flow",
                    "attack_steps",
                    "outcomes",
                ],
            )
            connections.disconnect("default")

            if not pattern_details:
                raise HTTPException(
                    status_code=404, detail=f"Patrón {pattern_id} no encontrado"
                )

            pattern = pattern_details[0]

            # Crear un contexto enriquecido con todos los detalles del patrón
            enriched_prompt = f"""
Analiza el siguiente patrón de ataque CAPEC {pattern_id} con todos sus detalles:

INFORMACIÓN BÁSICA:
- ID: {pattern.get('pattern_id')}
- Nombre: {pattern.get('name')}
- Estado: {pattern.get('status')}
- Nivel de Abstracción: {pattern.get('abstraction')}

DESCRIPCIÓN Y RESUMEN:
- Descripción: {pattern.get('description')}
- Resumen: {pattern.get('summary')}
- Términos Alternativos: {pattern.get('alternate_terms')}

METADATA:
- Fecha de Envío: {pattern.get('submission_date')}
- Autor: {pattern.get('submission_name')}
- Organización: {pattern.get('submission_organization')}

EVALUACIÓN DE RIESGO:
- Severidad Típica: {pattern.get('typical_severity')}
- Probabilidad de Ataque: {pattern.get('likelihood_of_attack')}

REQUISITOS TÉCNICOS:
- Prerrequisitos: {pattern.get('prerequisites')}
- Habilidades Requeridas: {pattern.get('skills_required')}
- Recursos Necesarios: {pattern.get('resources_required')}
- Indicadores: {pattern.get('indicators')}

IMPACTO Y MITIGACIÓN:
- Consecuencias: {pattern.get('consequences')}
- Mitigaciones: {pattern.get('mitigations')}

EJEMPLOS Y NOTAS:
- Instancias de Ejemplo: {pattern.get('example_instances')}
- Notas Adicionales: {pattern.get('notes')}

RELACIONES:
- Patrones de Ataque Relacionados: {pattern.get('related_attack_patterns')}
- Debilidades Relacionadas: {pattern.get('related_weaknesses')}
- Mapeos de Taxonomía: {pattern.get('taxonomy_mappings')}

FLUJO DE EJECUCIÓN:
- Pasos de Ejecución: {pattern.get('execution_flow')}
- Pasos del Ataque: {pattern.get('attack_steps')}
- Resultados: {pattern.get('outcomes')}

{prompt}
"""

        except Exception as e:
            logger.error(f"Error al buscar en Milvus: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Error al buscar el patrón: {str(e)}"
            )

        logger.info(f"{enriched_prompt}")
        response = ollama_client.chat(
            model="qwen2.5-coder:7b",
            messages=[
                {
                    "role": "system",
                    "content": """
                    Eres un experto en ciberseguridad especializado en analizar patrones de ataque CAPEC.
                    Proporciona análisis detallados y útiles, enfocándote en aspectos prácticos y aplicables.                    
                    """,
                },
                {"role": "user", "content": enriched_prompt},
            ],
            stream=True,
            options={"temperature": 0.7},
        )

        async def generate():
            try:
                global OLLAMA_PATTERN_RESPONSE
                OLLAMA_PATTERN_RESPONSE = ""  # Reset for new response
                for chunk in response:
                    if chunk and "message" in chunk and "content" in chunk["message"]:
                        content = chunk["message"]["content"]
                        OLLAMA_PATTERN_RESPONSE += content  # Accumulate response
                        yield f"data: {json.dumps({'response': content, 'pattern_id': pattern_id}, ensure_ascii=False)}\n"
            except Exception as e:
                logger.error(f"Error en el streaming del análisis: {str(e)}")
                yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    except Exception as e:
        logger.error(f"Error en el endpoint /ollama/analyze: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def execute_tool(tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a Kali Linux tool with the provided parameters."""
    if tool_name not in KALI_TOOLS:
        logger.warning(f"Unknown tool requested: {tool_name}")
        return {
            "status": "error",
            "message": f"Unknown tool: {tool_name}. Available tools: {', '.join(KALI_TOOLS.keys())}",
        }

    logger.info(f"Executing {tool_name} with params: {params}")

    try:
        # Forward the request to the Kali Linux API server
        response = requests.post(
            f"{KALI_API_BASE_URL}{KALI_TOOLS[tool_name]}",
            json=params,
            timeout=300,  # Some tools might take time to execute
        )

        # Parse and return the response
        if response.status_code == 200:
            try:
                result = response.json()
                logger.info(f"Tool {tool_name} executed successfully")
                return {"status": "success", "tool": tool_name, "results": result}
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing API response: {str(e)}")
                return {
                    "status": "error",
                    "message": f"Error parsing API response: {str(e)}",
                    "raw_response": response.text,
                }
        else:
            logger.error(
                f"Tool execution failed: {response.status_code} {response.text}"
            )
            return {
                "status": "error",
                "message": f"Tool execution failed: {response.text}",
                "code": response.status_code,
            }

    except requests.RequestException as e:
        logger.error(f"Error executing tool {tool_name}: {str(e)}")
        return {"status": "error", "message": f"Error executing tool: {str(e)}"}


def process_message(message: str) -> str:
    """
    Process a user message to detect and execute tool requests,
    otherwise pass to the LLM for normal conversation.
    """
    # Check if this looks like a tool request
    tool_request = extract_tool_request(message)

    if tool_request and "tool" in tool_request:
        tool_name = tool_request["tool"]
        params = tool_request["params"]

        # Execute the tool
        result = execute_tool(tool_name, params)

        # Format tool results for LLM
        if result["status"] == "success":
            tool_output = f"Tool: {tool_name}\nStatus: Success\n\n"

            if "results" in result and "stdout" in result["results"]:
                # Clean and format the output
                stdout = result["results"]["stdout"]
                tool_output += f"Output:\n```\n{stdout}\n```\n"

            # Add any errors
            if (
                "results" in result
                and "stderr" in result["results"]
                and result["results"]["stderr"]
            ):
                stderr = result["results"]["stderr"]
                tool_output += f"\nErrors/Warnings:\n```\n{stderr}\n```\n"

            return tool_output
        else:
            return f"Failed to execute {tool_name}: {result.get('message', 'Unknown error')}"

    # If no tool request detected, use Ollama for conversation
    return None


def extract_tool_request(message: str) -> Dict:
    """
    Extract tool name and parameters from the LLM message.
    This is a simple approach and can be enhanced with better parsing.
    """
    # Look for patterns like "Run nmap scan on 10.10.10.10"
    for tool in KALI_TOOLS.keys():
        if tool in message.lower():
            # Very basic parameter extraction - will need improvement
            params = {}
            # Common parameters for each tool
            if tool == "nmap":
                if "target" not in params and "10.10.10." in message:
                    # Extract IP-like patterns
                    import re

                    ip_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
                    ip_matches = re.findall(ip_pattern, message)
                    if ip_matches:
                        params["target"] = ip_matches[0]

            # For web tools, look for URLs
            if tool in ["gobuster", "dirb", "nikto", "wpscan", "sqlmap"]:
                if "url" not in params and "http" in message:
                    import re

                    url_pattern = r"https?://[^\s]+"
                    url_matches = re.findall(url_pattern, message)
                    if url_matches:
                        params["url"] = url_matches[0]

            # If we have valid params, return the tool request
            if params:
                return {"tool": tool, "params": params}
    return {}


def generate_kali_attack_prompt(pattern_id: str, target: str = None) -> Dict[str, Any]:
    """Generar un plan de ataque usando herramientas de Kali basado en el análisis de patrones CAPEC."""
    try:
        global OLLAMA_PATTERN_RESPONSE
        if not OLLAMA_PATTERN_RESPONSE:
            return {
                "status": "error",
                "message": "No hay análisis de patrones disponible",
            }

        attack_prompt = f"""
Basado en este análisis de patrones CAPEC:
{OLLAMA_PATTERN_RESPONSE}

Genera un plan de ataque específico usando herramientas de Kali Linux contra: {target if target else 'example.com'}

Para este patrón {pattern_id}, proporciona:
1. Una lista de herramientas de Kali recomendadas de estas opciones disponibles: {', '.join(KALI_TOOLS.keys())}
2. Los comandos exactos a ejecutar con estas herramientas contra el objetivo {target if target else 'example.com'}
3. Los resultados esperados e indicadores de éxito
4. Cualquier prerrequisito o configuración necesaria
5. Los pasos del ataque en orden

Formatea tu respuesta como un plan estructurado con secciones claras y ejemplos de comandos.
Solo incluye herramientas de la lista disponible: {', '.join(KALI_TOOLS.keys())}
Asegúrate de que todos los comandos sean prácticos y ejecutables contra el objetivo especificado.
"""
        logger.info(f"Generado el prompt de ataque para el objetivo: {target}")
        # Get attack plan from Ollama
        response = ollama_client.chat(
            model="qwen2.5-coder:7b",
            messages=[
                {
                    "role": "system",
                    "content": "Eres un experto en pruebas de penetración especializado en herramientas de Kali Linux y patrones de ataque CAPEC. Proporciona planes de ataque prácticos y específicos.",
                },
                {"role": "user", "content": attack_prompt},
            ],
            options={"temperature": 0.7},
        )

        if response and "message" in response and "content" in response["message"]:
            return {
                "status": "success",
                "attack_plan": response["message"]["content"],
                "pattern_id": pattern_id,
            }
        else:
            return {
                "status": "error",
                "message": "No se pudo generar el plan de ataque",
            }

    except Exception as e:
        logger.error(f"Error generando el plan de ataque: {str(e)}")
        return {"status": "error", "message": str(e)}


@app.post("/ollama/generate_attack_plan/{pattern_id}")
async def get_attack_plan(pattern_id: str, target: str = Query(None)):
    """Endpoint para generar un plan de ataque de Kali Linux basado en un patrón CAPEC."""
    try:
        if not pattern_id:
            raise HTTPException(status_code=400, detail="Se requiere el ID del patrón")

        async def generate():
            try:
                global OLLAMA_PATTERN_RESPONSE
                if not OLLAMA_PATTERN_RESPONSE:
                    yield f"data: {json.dumps({'error': 'No hay análisis de patrones disponible'})}\n\n"
                    return

                attack_prompt = f"""
Basado en este análisis de patrones CAPEC:
{OLLAMA_PATTERN_RESPONSE}

Genera un plan de ataque específico usando herramientas de Kali Linux contra: {target if target else 'example.com'}

Para este patrón {pattern_id}, proporciona:
1. Una lista de herramientas de Kali recomendadas de estas opciones disponibles: {', '.join(KALI_TOOLS.keys())}
2. Los comandos exactos a ejecutar con estas herramientas contra el objetivo {target if target else 'example.com'}
3. Los resultados esperados e indicadores de éxito
4. Cualquier prerrequisito o configuración necesaria
5. Los pasos del ataque en orden

Formatea tu respuesta como un plan estructurado con secciones claras y ejemplos de comandos.
Solo incluye herramientas de la lista disponible: {', '.join(KALI_TOOLS.keys())}
Asegúrate de que todos los comandos sean prácticos y ejecutables contra el objetivo especificado.
"""
                # Get attack plan from Ollama with streaming
                stream = ollama_client.chat(
                    model="qwen2.5-coder:7b",
                    messages=[
                        {
                            "role": "system",
                            "content": "Eres un experto en pruebas de penetración especializado en herramientas de Kali Linux y patrones de ataque CAPEC. Proporciona planes de ataque prácticos y específicos.",
                        },
                        {"role": "user", "content": attack_prompt},
                    ],
                    stream=True,
                    options={"temperature": 0.7},
                )

                for response in stream:
                    if (
                        response
                        and "message" in response
                        and "content" in response["message"]
                    ):
                        content = response["message"]["content"]
                        yield f"data: {json.dumps({'response': content}, ensure_ascii=False)}\n\n"

            except Exception as e:
                logger.error(f"Error en la generación del plan de ataque: {str(e)}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "sin caché",
                "Connection": "mantener viva",
            },
        )

    except Exception as e:
        logger.error(f"Error en el endpoint del plan de ataque: {str(e)}")
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
        log_level="info",
    )
