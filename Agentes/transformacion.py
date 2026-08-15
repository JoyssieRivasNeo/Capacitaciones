import logging
import os
import re
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

from .gemini_utils import generar_con_modelos

load_dotenv(Path(__file__).parent / ".env")

logger = logging.getLogger(__name__)

# Cliente y API key dedicados a transcripción, separados de los usados para
# responder preguntas (GEMINI_API_KEY) — así cargar un video no compite por
# cuota gratuita con los usuarios haciendo preguntas al mismo tiempo.
MODELOS_TRANSCRIPCION = ["gemini-flash-latest", "gemini-flash-lite-latest"]

_PATRON_VIDEO_ID = re.compile(
    r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})"
)

_cliente_videos: Optional[genai.Client] = None


def extraer_video_id(url_o_id: str) -> str:
    """Obtiene el video_id a partir de una URL de YouTube (o lo devuelve tal cual si ya es un ID)."""
    match = _PATRON_VIDEO_ID.search(url_o_id)
    return match.group(1) if match else url_o_id


def _obtener_cliente_videos() -> genai.Client:
    """Devuelve (creando si hace falta) el cliente de Gemini para transcripción, con su propia API key."""
    global _cliente_videos
    if _cliente_videos is None:
        _cliente_videos = genai.Client(api_key=os.environ["GEMINI_API_KEY_VIDEOS"])
    return _cliente_videos


def obtener_transcripcion_segura(url_o_id: str) -> Optional[str]:
    """
    Devuelve la transcripción de un video de YouTube a partir de su URL o video_id.

    Usa la capacidad de Gemini de procesar video directamente por URL: la
    petición a YouTube la hace la infraestructura interna de Google, no
    nuestro servidor, así que evita el bloqueo de IPs de datacenter que
    sufren las librerías de scraping (como youtube_transcript_api) al correr
    en la nube (ej. Railway). Devuelve None si no se pudo transcribir con
    ningún modelo de `MODELOS_TRANSCRIPCION`.
    """
    video_id = extraer_video_id(url_o_id)
    cliente = _obtener_cliente_videos()

    contents = types.Content(parts=[
        types.Part(file_data=types.FileData(file_uri=f"https://youtu.be/{video_id}")),
        types.Part(text=(
            "Transcribe el audio de este video completo, palabra por palabra, sin "
            "resumir ni omitir contenido. Incluye el nombre del hablante antes de "
            "su intervención si es claramente distinguible (formato 'Nombre: texto'). "
            "Devuelve texto plano únicamente: sin marcas de tiempo, sin encabezados, "
            "sin formato markdown (nada de negritas, títulos ni líneas separadoras '---'), "
            "y sin describir sonidos, música o silencios entre paréntesis o corchetes."
        )),
    ])

    texto = generar_con_modelos(cliente, MODELOS_TRANSCRIPCION, contents)
    if texto is None:
        logger.error("No se pudo transcribir el video %s con ningún modelo de Gemini.", video_id)
    return texto
