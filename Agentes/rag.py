import logging
import time
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv
from google import genai
from google.genai.errors import ClientError, ServerError

from .savedatabase import buscar_semantica

load_dotenv(Path(__file__).parent / ".env")

logger = logging.getLogger(__name__)

# Se intentan en orden: si el primero está saturado tras los reintentos, se
# prueba con el siguiente (cada modelo tiene su propia cuota/capacidad).
MODELOS_GEMINI = ["gemini-flash-latest", "gemini-flash-lite-latest"]
MAX_REINTENTOS = 3
ESPERA_INICIAL = 2.0

_cliente: Optional[genai.Client] = None
_cache_respuestas: Dict[str, str] = {}


def _obtener_cliente() -> genai.Client:
    """Devuelve (creando si hace falta) el cliente de Gemini, leyendo GEMINI_API_KEY del entorno."""
    global _cliente
    if _cliente is None:
        _cliente = genai.Client()
    return _cliente


def _generar_con_reintentos(cliente: genai.Client, modelo: str, prompt: str) -> Optional[str]:
    """
    Llama a `modelo`. Ante rate limiting (429) reintenta con backoff
    exponencial en el mismo modelo. Ante servidor sobrecargado (5xx) no
    insiste con el mismo modelo — devuelve None de inmediato para que el
    llamador pruebe el siguiente modelo de respaldo sin demora.
    """
    espera = ESPERA_INICIAL

    for intento in range(1, MAX_REINTENTOS + 1):
        try:
            respuesta = cliente.models.generate_content(model=modelo, contents=prompt)
            return respuesta.text

        except ServerError as e:
            logger.warning(
                "%s sobrecargado (código %s). Probando el siguiente modelo.",
                modelo, e.code,
            )
            return None

        except ClientError as e:
            if e.code != 429:
                logger.exception("Error del cliente al consultar Gemini (%s).", modelo)
                return None
            if intento == MAX_REINTENTOS:
                logger.warning(
                    "Modelo %s no respondió tras %d intentos (rate limit).",
                    modelo, MAX_REINTENTOS,
                )
                return None
            logger.warning(
                "Rate limit en %s (intento %d/%d). Reintentando en %.0fs...",
                modelo, intento, MAX_REINTENTOS, espera,
            )
            time.sleep(espera)
            espera *= 2

    return None


def responder_pregunta(pregunta: str, video_id: Optional[str] = None, n_chunks: int = 3) -> str:
    """
    Responde una pregunta con RAG: busca los chunks más relevantes en la base
    vectorial y le pide a Gemini que responda basándose solo en ese contexto.

    Si se pasa `video_id`, la búsqueda se limita a ese video; si se omite,
    busca en todos los videos guardados. Cachea respuestas por pregunta +
    video (reduce peticiones repetidas al límite gratuito). Ante rate
    limiting (429) reintenta con backoff en el mismo modelo; ante servidor
    sobrecargado (5xx) pasa de inmediato al siguiente modelo de
    `MODELOS_GEMINI`, sin esperar.
    """
    clave_cache = f"{video_id or 'todos'}::{pregunta.strip().lower()}"
    if clave_cache in _cache_respuestas:
        logger.info("Respuesta servida desde caché para: %s", pregunta)
        return _cache_respuestas[clave_cache]

    chunks = buscar_semantica(pregunta, n_results=n_chunks, video_id=video_id)
    if not chunks:
        return "No encontré información relevante para responder esa pregunta."

    contexto = "\n\n".join(chunks)
    prompt = (
        "Eres un asistente que responde preguntas sobre capacitaciones (webinars) de Google Analytics "
        "a partir de transcripciones automáticas de YouTube. Esas transcripciones incluyen ruido que no "
        "es parte del contenido educativo: música y letras de canciones de introducción, aplausos, "
        "risas, pausas y otras interjecciones. Ignora por completo ese ruido — incluso si no está entre "
        "corchetes — y responde basándote únicamente en el contenido real del webinar (explicaciones, "
        "casos de estudio, datos, pasos, etc.).\n\n"
        "Responde la siguiente pregunta usando únicamente la información relevante del contexto. "
        "Si el contexto no contiene la respuesta, dilo explícitamente en vez de inventar.\n\n"
        f"Contexto:\n{contexto}\n\nPregunta: {pregunta}"
    )

    cliente = _obtener_cliente()

    for modelo in MODELOS_GEMINI:
        texto = _generar_con_reintentos(cliente, modelo, prompt)
        if texto is not None:
            _cache_respuestas[clave_cache] = texto
            return texto

    logger.error("Todos los modelos de Gemini fallaron para la pregunta: %s", pregunta)
    return "El servicio está saturado en este momento, por favor intenta de nuevo en unos minutos."
