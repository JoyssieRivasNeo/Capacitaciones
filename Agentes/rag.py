import logging
import os
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv
from google import genai

from .gemini_utils import generar_con_modelos
from .savedatabase import buscar_semantica

load_dotenv(Path(__file__).parent / ".env")

logger = logging.getLogger(__name__)

# Se intentan en orden: si el primero está saturado, se prueba con el
# siguiente (cada modelo tiene su propia cuota/capacidad).
MODELOS_GEMINI = ["gemini-flash-latest", "gemini-flash-lite-latest"]

_cliente: Optional[genai.Client] = None
_cache_respuestas: Dict[str, str] = {}


def _obtener_cliente() -> genai.Client:
    """Devuelve (creando si hace falta) el cliente de Gemini, leyendo GEMINI_API_KEY_RSPTA del entorno."""
    global _cliente
    if _cliente is None:
        _cliente = genai.Client(api_key=os.environ["GEMINI_API_KEY_RSPTA"])
    return _cliente


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
        "a partir de transcripciones automáticas de YouTube. Esas transcripciones pueden incluir ruido "
        "que no es parte del contenido educativo: música y letras de canciones de introducción, "
        "aplausos, risas, pausas y otras interjecciones. Ignora por completo ese ruido si aparece — "
        "incluso si no está entre corchetes — y responde basándote únicamente en el contenido real del "
        "webinar (explicaciones, casos de estudio, datos, pasos, etc.).\n\n"
        "Responde la siguiente pregunta usando únicamente la información relevante del contexto. "
        "Si el contexto no contiene la respuesta, dilo explícitamente en vez de inventar.\n\n"
        f"Contexto:\n{contexto}\n\nPregunta: {pregunta}"
    )

    cliente = _obtener_cliente()
    texto = generar_con_modelos(cliente, MODELOS_GEMINI, prompt)

    if texto is not None:
        _cache_respuestas[clave_cache] = texto
        return texto

    logger.error("Todos los modelos de Gemini fallaron para la pregunta: %s", pregunta)
    return "El servicio está saturado en este momento, por favor intenta de nuevo en unos minutos."
