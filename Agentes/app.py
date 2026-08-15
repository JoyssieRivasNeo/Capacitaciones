import logging
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .rag import responder_pregunta
from .savedatabase import eliminar_video, guardar_video_en_vector_db, listar_videos

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Chatbot de Capacitaciones")

DIRECTORIO_ESTATICO = Path(__file__).parent / "static"

app.mount("/static", StaticFiles(directory=DIRECTORIO_ESTATICO), name="static")


class PreguntaRequest(BaseModel):
    pregunta: str
    video_id: Optional[str] = None


class VideoRequest(BaseModel):
    url: str
    nombre: Optional[str] = None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(DIRECTORIO_ESTATICO / "index.html")


@app.post("/api/preguntar")
def preguntar(datos: PreguntaRequest) -> dict:
    if not datos.pregunta.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")
    respuesta = responder_pregunta(datos.pregunta, video_id=datos.video_id)
    return {"respuesta": respuesta}


@app.post("/api/videos")
def agregar_video(datos: VideoRequest) -> dict:
    if not datos.url.strip():
        raise HTTPException(status_code=400, detail="La URL no puede estar vacía.")

    chunks_guardados = guardar_video_en_vector_db(datos.url, nombre=datos.nombre)
    if chunks_guardados == 0:
        raise HTTPException(
            status_code=422,
            detail="No se pudo procesar el video (sin transcripción disponible).",
        )
    return {"mensaje": "Video procesado correctamente.", "chunks_guardados": chunks_guardados}


@app.get("/api/videos")
def obtener_videos() -> List[dict]:
    return listar_videos()


@app.delete("/api/videos/{video_id}")
def borrar_video(video_id: str) -> dict:
    eliminados = eliminar_video(video_id)
    if eliminados == 0:
        raise HTTPException(status_code=404, detail="No se encontró ese video en la base.")
    return {"mensaje": "Video eliminado.", "chunks_eliminados": eliminados}
