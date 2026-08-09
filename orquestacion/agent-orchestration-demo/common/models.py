"""
Modelos de datos compartidos.

Estas dataclasses son el "contrato" que agentes, orquestadores, router
y aggregator usan para intercambiar información. Mantenerlas simples es
justamente lo que permite que todo el proyecto sea fácil de leer.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentRequest:
    """Solicitud original que dispara todo el flujo (pipeline o router)."""

    text: str
    context: dict = field(default_factory=dict)


@dataclass
class AgentResult:
    """Resultado producido por un agente al ejecutar `run()`."""

    agent: str
    status: str
    answer: Any
    metadata: dict = field(default_factory=dict)


@dataclass
class RoutingDecision:
    """Decisión tomada por el RouterAgent: qué agentes deben ejecutarse."""

    selected_agents: list[str]
    reason: str


@dataclass
class FinalResponse:
    """Respuesta final que se entrega al usuario tras la agregación."""

    query: str
    agents_used: list[str]
    answer: str
    details: list[AgentResult] = field(default_factory=list)
