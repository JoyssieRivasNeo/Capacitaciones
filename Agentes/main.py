import logging

from .rag import responder_pregunta
from .savedatabase import guardar_video_en_vector_db


def main() -> None:
    video_id = "ysVKi5BXTmE"
    guardar_video_en_vector_db(video_id)

    pregunta = "¿Cómo optimizar los títulos de los productos en Merchant Center?"
    respuesta = responder_pregunta(pregunta)

    print("\n--- Respuesta ---")
    print(respuesta)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
