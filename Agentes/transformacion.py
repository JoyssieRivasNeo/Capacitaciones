import logging
import re
import time
from typing import Optional

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)
from youtube_transcript_api._errors import RequestBlocked

logger = logging.getLogger(__name__)

_PATRON_VIDEO_ID = re.compile(
    r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})"
)

# Fragmentos que son en su totalidad una anotación no verbal de subtítulos
# automáticos, ej. "[Music]", "[Applause]", "[Risas]", "(música)".
_PATRON_ANOTACION_NO_VERBAL = re.compile(r"^[\[\(].*[\]\)]$")


def extraer_video_id(url_o_id: str) -> str:
    """Obtiene el video_id a partir de una URL de YouTube (o lo devuelve tal cual si ya es un ID)."""
    match = _PATRON_VIDEO_ID.search(url_o_id)
    return match.group(1) if match else url_o_id


def _es_anotacion_no_verbal(texto: str) -> bool:
    """True si el fragmento es solo una anotación de sonido (no habla), ej. "[Music]"."""
    return bool(_PATRON_ANOTACION_NO_VERBAL.match(texto.strip()))


def obtener_transcripcion_segura(
    url_o_id: str,
    idiomas_preferidos: Optional[list[str]] = None,
    idioma_fallback: str = "en",
    idioma_traduccion: str = "es",
    max_reintentos: int = 3,
    espera_inicial: float = 2.0,
) -> Optional[str]:
    """
    Devuelve la transcripción de un video de YouTube a partir de su URL o video_id.

    Busca subtítulos (manuales o automáticos) en `idiomas_preferidos`. Si no
    existen, intenta traducir automáticamente los de `idioma_fallback` al
    `idioma_traduccion`. Si YouTube bloquea las peticiones (rate limiting),
    reintenta con backoff exponencial hasta `max_reintentos` veces. Devuelve
    None si no hay transcripción disponible o si fallan todos los intentos.
    """
    idiomas_preferidos = idiomas_preferidos or ["es", "en"]
    video_id = extraer_video_id(url_o_id)
    espera = espera_inicial
    api = YouTubeTranscriptApi()

    for intento in range(1, max_reintentos + 1):
        try:
            transcript_list = api.list(video_id)

            try:
                transcript = transcript_list.find_transcript(idiomas_preferidos)
            except NoTranscriptFound:
                transcript = transcript_list.find_transcript(
                    [idioma_fallback]
                ).translate(idioma_traduccion)

            return "\n".join(
                snippet.text
                for snippet in transcript.fetch()
                if not _es_anotacion_no_verbal(snippet.text)
            )

        except RequestBlocked:
            if intento == max_reintentos:
                logger.error(
                    "YouTube bloqueó las peticiones para %s tras %d intentos.",
                    video_id, max_reintentos,
                )
                return None
            logger.warning(
                "Rate limit de YouTube para %s (intento %d/%d). Reintentando en %.0fs...",
                video_id, intento, max_reintentos, espera,
            )
            time.sleep(espera)
            espera *= 2

        except TranscriptsDisabled:
            logger.error("El video %s tiene los subtítulos desactivados.", video_id)
            return None
        except NoTranscriptFound:
            logger.error("No se encontró ningún subtítulo para %s.", video_id)
            return None
        except VideoUnavailable:
            logger.error("El video %s es privado o no existe.", video_id)
            return None
        except Exception:
            logger.exception("Error imprevisto al procesar %s", video_id)
            return None

    return None
