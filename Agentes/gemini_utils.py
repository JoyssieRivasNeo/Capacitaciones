import logging
import time
from typing import List, Optional, Union

from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError

logger = logging.getLogger(__name__)

MAX_REINTENTOS = 3
ESPERA_INICIAL = 2.0

Contenido = Union[str, types.Content]


def generar_con_reintentos(cliente: genai.Client, modelo: str, contents: Contenido) -> Optional[str]:
    """
    Llama a `modelo` con `contents` (texto simple o contenido multimodal).

    Ante rate limiting (429) reintenta con backoff exponencial en el mismo
    modelo. Ante servidor sobrecargado (5xx) no insiste con el mismo modelo
    — devuelve None de inmediato para que el llamador pruebe el siguiente
    modelo de respaldo sin demora.
    """
    espera = ESPERA_INICIAL

    for intento in range(1, MAX_REINTENTOS + 1):
        try:
            respuesta = cliente.models.generate_content(model=modelo, contents=contents)
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


def generar_con_modelos(cliente: genai.Client, modelos: List[str], contents: Contenido) -> Optional[str]:
    """Prueba cada modelo de `modelos` en orden hasta obtener una respuesta exitosa."""
    for modelo in modelos:
        texto = generar_con_reintentos(cliente, modelo, contents)
        if texto is not None:
            return texto
    return None
