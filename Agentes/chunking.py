import logging
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .transformacion import obtener_transcripcion_segura

logger = logging.getLogger(__name__)


def extraer_y_fragmentar(
    url_o_id: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[str]:
    """
    Obtiene la transcripción de un video de YouTube y la fragmenta en chunks.

    chunk_size: Tamaño máximo de cada bloque (en caracteres)
    chunk_overlap: Caracteres de superposición entre bloques para no perder contexto
    """
    texto_completo = obtener_transcripcion_segura(url_o_id)
    if not texto_completo:
        logger.error("No se pudo obtener la transcripción de %s, no se generarán chunks.", url_o_id)
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],  # Intenta cortar en este orden de prioridad
    )

    chunks = text_splitter.split_text(texto_completo)
    logger.info("Texto procesado con éxito. Se generaron %d chunks.", len(chunks))
    return chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    video_id = "ysVKi5BXTmE"
    mis_chunks = extraer_y_fragmentar(video_id)

    if mis_chunks:
        print("\n--- CHUNK 1 DE MUESTRA ---")
        print(mis_chunks[0])
