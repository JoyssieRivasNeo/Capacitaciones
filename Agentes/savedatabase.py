import logging
from typing import Any, Dict, List, Optional

import chromadb

from .chunking import extraer_y_fragmentar
from .transformacion import extraer_video_id

logger = logging.getLogger(__name__)

RUTA_BD = "./mi_base_de_datos_yt"
NOMBRE_COLECCION = "transcripciones_youtube"

_cliente: Optional[chromadb.PersistentClient] = None


def obtener_coleccion() -> Any:
    """Devuelve (creando si hace falta) la colección de ChromaDB usada para guardar transcripciones."""
    global _cliente
    if _cliente is None:
        _cliente = chromadb.PersistentClient(path=RUTA_BD)
    return _cliente.get_or_create_collection(name=NOMBRE_COLECCION)


def guardar_video_en_vector_db(
    url_o_id: str,
    nombre: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
) -> int:
    """
    Extrae, fragmenta y guarda (o actualiza) los chunks de un video en la base vectorial.

    Acepta una URL de YouTube o un video_id; internamente siempre se normaliza
    al ID limpio antes de guardar, para que los ids/metadatos en la BD queden
    consistentes sin importar el formato de entrada. `nombre` es una etiqueta
    legible para identificar el video en la interfaz (si se omite, se usa el
    video_id). Usa upsert, así que puede ejecutarse varias veces sobre el
    mismo video sin fallar por ids duplicados. Devuelve la cantidad de chunks
    guardados.
    """
    video_id = extraer_video_id(url_o_id)
    nombre = nombre.strip() if nombre and nombre.strip() else video_id

    chunks = extraer_y_fragmentar(video_id, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        logger.error("No se generaron chunks para %s, no se guardó nada en la BD.", video_id)
        return 0

    ids = [f"{video_id}_chunk_{i}" for i in range(len(chunks))]
    metadatos = [
        {"video_id": video_id, "chunk_index": i, "nombre": nombre}
        for i in range(len(chunks))
    ]

    coleccion = obtener_coleccion()
    # ChromaDB genera los embeddings automáticamente si no se le pasa un modelo específico
    coleccion.upsert(documents=chunks, metadatas=metadatos, ids=ids)

    logger.info("Guardados %d chunks del video %s ('%s') en la BD.", len(chunks), video_id, nombre)
    return len(chunks)


def listar_videos() -> List[Dict[str, Any]]:
    """Devuelve un resumen (video_id, nombre, cantidad de chunks) de cada video guardado en la BD."""
    coleccion = obtener_coleccion()
    resultados = coleccion.get(include=["metadatas"])

    resumen: Dict[str, Dict[str, Any]] = {}
    for metadato in resultados["metadatas"]:
        video_id = metadato["video_id"]
        if video_id not in resumen:
            resumen[video_id] = {
                "video_id": video_id,
                "nombre": metadato.get("nombre", video_id),
                "chunks": 0,
            }
        resumen[video_id]["chunks"] += 1

    return sorted(resumen.values(), key=lambda v: v["nombre"].lower())


def eliminar_video(video_id: str) -> int:
    """Elimina todos los chunks de `video_id` de la base vectorial. Devuelve cuántos se borraron."""
    coleccion = obtener_coleccion()
    existentes = coleccion.get(where={"video_id": video_id}, include=[])
    cantidad = len(existentes["ids"])

    if cantidad > 0:
        coleccion.delete(where={"video_id": video_id})
        logger.info("Eliminados %d chunks del video %s.", cantidad, video_id)

    return cantidad


def buscar_semantica(query: str, n_results: int = 2, video_id: Optional[str] = None) -> List[str]:
    """
    Busca los chunks más relevantes para `query` dentro de la base vectorial.

    Si se pasa `video_id`, la búsqueda se limita a los chunks de ese video
    únicamente; si se omite, busca en todos los videos guardados.
    """
    coleccion = obtener_coleccion()
    where = {"video_id": video_id} if video_id else None
    resultados = coleccion.query(query_texts=[query], n_results=n_results, where=where)
    return resultados["documents"][0]
