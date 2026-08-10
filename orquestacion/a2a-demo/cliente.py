"""
Cliente A2A usando el SDK OFICIAL (a2a-sdk) - no arma el JSON-RPC a mano.

Dos pasos, que son el corazon de A2A:
  1. Descubrimiento: A2ACardResolver pide el Agent Card (que sabe hacer el
     agente, a que URL hablarle). El cliente no necesita importar ningun
     codigo del servidor — solo su URL.
  2. Invocacion: ClientFactory arma un Client "hablando" el transporte que
     declara el Agent Card (JSONRPC por defecto), y client.send_message()
     manda el mensaje.

Nota: send_message() siempre devuelve un ITERABLE de eventos (async for),
aunque el agente responda una sola vez — eso es lo que le permite al mismo
metodo servir tanto respuestas inmediatas (un evento) como Tasks largas que
van emitiendo actualizaciones (varios eventos).

Uso (con agente_servidor.py corriendo en otra terminal):
    python cliente.py "hola mundo gracias por favor"
    python cliente.py                # usa un mensaje de ejemplo
"""
from __future__ import annotations

import asyncio
import sys
import uuid

import httpx

from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import AgentCard, Message, Part, Role, TextPart

SERVIDOR = "http://127.0.0.1:9000"


async def descubrir_agente(httpx_client: httpx.AsyncClient) -> AgentCard:
    """Paso 1: lee el Agent Card, el 'menu' publico del agente."""
    resolver = A2ACardResolver(httpx_client, base_url=SERVIDOR)
    card = await resolver.get_agent_card()
    print(f"[cliente] Agent Card descubierta: {card.name} - {card.description}")
    for skill in card.skills:
        print(f"[cliente]   skill: {skill.id} ({skill.description})")
    return card


async def enviar_mensaje(card: AgentCard, httpx_client: httpx.AsyncClient, texto: str) -> str:
    """Paso 2: manda el texto como Message y junta la respuesta del agente."""
    factory = ClientFactory(ClientConfig(httpx_client=httpx_client, streaming=False))
    client = factory.create(card)

    mensaje = Message(
        role=Role.user,
        message_id=str(uuid.uuid4()),
        parts=[Part(root=TextPart(text=texto))],
    )

    print(f'[cliente] Enviando mensaje -> "{texto}"')
    async for evento in client.send_message(mensaje):
        # Este agente responde inmediato: el primer evento ya es la respuesta.
        if isinstance(evento, Message):
            return "".join(p.root.text for p in evento.parts if hasattr(p.root, "text"))
        # Si el agente fuera de tipo Task, evento seria (Task, update) en su lugar.
        raise RuntimeError(f"Se esperaba un Message inmediato, llego: {evento!r}")

    raise RuntimeError("El agente no devolvio ninguna respuesta.")


async def main_async() -> None:
    texto = " ".join(sys.argv[1:]) or "hola mundo gracias"
    async with httpx.AsyncClient() as httpx_client:
        try:
            card = await descubrir_agente(httpx_client)
        except httpx.ConnectError:
            print("[error] No se pudo conectar al servidor A2A. "
                  "Corre primero: python agente_servidor.py")
            sys.exit(1)

        resultado = await enviar_mensaje(card, httpx_client, texto)
        print(f"\n--- Resultado ---\n{resultado}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
