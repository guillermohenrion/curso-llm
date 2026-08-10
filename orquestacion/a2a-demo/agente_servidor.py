"""
Servidor A2A usando el SDK OFICIAL (a2a-sdk, del proyecto Agent2Agent /
Linux Foundation) - no reimplementa el protocolo a mano.

A2A (https://a2a-protocol.org) es un protocolo abierto para que agentes
construidos con frameworks o proveedores DISTINTOS se hablen entre si por
HTTP, sin conocer el codigo interno del otro. Este archivo solo escribe
la LOGICA del agente; todo lo protocolar (Agent Card, JSON-RPC, Task,
validacion) lo arma el SDK:

    AgentExecutor          -> la interfaz que implementamos: execute() y
                               cancel(). Es el unico lugar con logica propia.
    DefaultRequestHandler   -> conecta el AgentExecutor con el transporte
                               (maneja el ciclo de vida de la Task).
    InMemoryTaskStore        -> guarda las Tasks en memoria (para una demo
                               alcanza; en produccion seria una DB).
    A2AStarletteApplication  -> arma la app ASGI: expone el Agent Card en
                               /.well-known/agent-card.json y el endpoint
                               JSON-RPC en "/".

Este agente responde de forma INMEDIATA (una sola vez, sin Task de fondo):
segun el spec, eso significa encolar un unico Message en vez de una Task
con estados intermedios - por eso execute() nunca crea un TaskUpdater.

Requisitos (ver requirements-a2a.txt):
    pip install -r requirements-a2a.txt

Uso:
    python agente_servidor.py           # levanta en http://127.0.0.1:9000
    python cliente.py                   # (en otra terminal) le habla por A2A
"""
from __future__ import annotations

import uvicorn

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, UnsupportedOperationError
from a2a.utils import new_agent_text_message
from a2a.utils.errors import ServerError

HOST, PORT = "127.0.0.1", 9000
AGENT_URL = f"http://{HOST}:{PORT}/"

# --- "Inteligencia" del agente: deliberadamente simple y deterministica ---
# (el foco de este ejemplo es el protocolo, no el traductor en si)
DICCIONARIO = {
    "hola": "hello", "mundo": "world", "gracias": "thank you",
    "buenos": "good", "dias": "days", "como": "how", "estas": "are you",
    "adios": "goodbye", "por": "for", "favor": "favor",
}


def traducir(texto: str) -> str:
    """Traduccion palabra por palabra contra un vocabulario chico de demo."""
    palabras = texto.lower().strip().split()
    return " ".join(DICCIONARIO.get(p.strip(".,?!"), f"[{p}]") for p in palabras)


class AgenteTraductor(AgentExecutor):
    """La unica parte de este archivo que NO es protocolo: la logica del agente."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        texto_entrada = context.get_user_input()
        print(f'[servidor] mensaje recibido -> "{texto_entrada}"')

        respuesta = traducir(texto_entrada)
        print(f'[servidor] respondiendo -> "{respuesta}"')

        # Respuesta inmediata: encolamos UN Message (no una Task de fondo).
        await event_queue.enqueue_event(new_agent_text_message(respuesta))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Esta demo responde siempre al toque; no hay nada largo que cancelar.
        raise ServerError(error=UnsupportedOperationError())


# --- Agent Card: lo primero que lee cualquier cliente A2A ---
AGENT_CARD = AgentCard(
    name="traductor_a2a",
    description="Traduce frases simples de espanol a ingles (vocabulario de demo).",
    url=AGENT_URL,
    version="0.1.0",
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain"],
    capabilities=AgentCapabilities(streaming=False),
    skills=[
        AgentSkill(
            id="traducir_es_en",
            name="Traducir espanol a ingles",
            description="Traduce palabra por palabra usando un vocabulario chico de demo.",
            tags=["traduccion", "demo"],
            examples=["hola mundo", "buenos dias"],
        )
    ],
)


def construir_app():
    """Arma la app ASGI: el SDK conecta Agent Card + JSON-RPC + nuestro AgentExecutor."""
    handler = DefaultRequestHandler(
        agent_executor=AgenteTraductor(),
        task_store=InMemoryTaskStore(),
    )
    return A2AStarletteApplication(agent_card=AGENT_CARD, http_handler=handler).build()


def main() -> None:
    print(f"[servidor] Agent Card en http://{HOST}:{PORT}/.well-known/agent-card.json")
    print(f"[servidor] Escuchando JSON-RPC en {AGENT_URL} (Ctrl+C para salir)")
    uvicorn.run(construir_app(), host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
