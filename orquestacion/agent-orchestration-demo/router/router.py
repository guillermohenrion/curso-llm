"""
AgentRegistry + RouterAgent
============================
El AgentRegistry es el "catálogo" de agentes disponibles: qué agentes
existen, qué hacen y qué palabras clave (capabilities) saben resolver.
El RouterAgent SOLO conoce este catálogo, nunca conoce el código interno
de cada agente.

Estrategia de enrutamiento (v1): keyword matching.

    normalizar(solicitud) -> buscar substrings de `capabilities`
    de cada agente del registro -> devolver los agentes que matchean.

Esta es la estrategia MÁS SIMPLE posible. El README explica cómo
evolucionar esto hacia un router basado en LLM + embeddings + RAG.
"""

from common.models import RoutingDecision

AGENT_REGISTRY = [
    {
        "name": "credit_agent",
        "description": "Especialista en créditos y préstamos",
        "capabilities": ["prestamo", "credito", "financiacion", "tasa"],
    },
    {
        "name": "investment_agent",
        "description": "Especialista en inversiones",
        "capabilities": ["inversion", "bono", "accion", "fondo"],
    },
    {
        "name": "card_agent",
        "description": "Especialista en tarjetas",
        "capabilities": ["tarjeta", "limite", "consumo", "bloqueo"],
    },
]

_ACCENTS = str.maketrans("áéíóúÁÉÍÓÚñÑ", "aeiouAEIOUnN")


def _normalize(text: str) -> str:
    return text.translate(_ACCENTS).lower()


class RouterAgent:
    """Decide qué agente(s) especializado(s) deben resolver la solicitud."""

    def __init__(self, registry: list[dict] = None):
        self.registry = registry if registry is not None else AGENT_REGISTRY

    def route(self, user_request: str) -> RoutingDecision:
        normalized_request = _normalize(user_request)

        selected_agents = []
        matched_keywords = {}

        for agent in self.registry:
            hits = [
                capability
                for capability in agent["capabilities"]
                if capability in normalized_request
            ]
            if hits:
                selected_agents.append(agent["name"])
                matched_keywords[agent["name"]] = hits

        if not selected_agents:
            return RoutingDecision(
                selected_agents=[],
                reason=(
                    "No se encontraron palabras clave que coincidan con "
                    "ningún agente del catálogo."
                ),
            )

        reason_parts = [
            f"{name} (por: {', '.join(keywords)})"
            for name, keywords in matched_keywords.items()
        ]
        reason = "Coincidencias encontradas -> " + "; ".join(reason_parts)

        return RoutingDecision(selected_agents=selected_agents, reason=reason)
