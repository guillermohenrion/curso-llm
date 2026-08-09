"""
Agentes especialistas del router.

Cada agente especialista sabe resolver un tipo de consulta muy concreto
(créditos, inversiones o tarjetas) y no sabe nada sobre los demás
agentes ni sobre cómo fue seleccionado: eso es responsabilidad del
RouterAgent (ver router/router.py).

Todos comparten la misma forma de trabajar (armar un prompt, llamar al
LLM, medir tiempo y devolver un AgentResult), por eso esa parte común
vive en `SpecialistAgent`. Lo único que cambia entre agentes es su
`name`, `description`, `capabilities` y `PROMPT`.
"""

import time

from common.models import AgentResult


class SpecialistAgent:
    """Base común para los agentes especialistas del router."""

    name = "base_agent"
    description = "Agente base (no se usa directamente)."
    capabilities: list[str] = []
    PROMPT = ""

    def __init__(self, llm):
        self.llm = llm
        self.last_prompt = ""

    def run(self, user_request: str) -> AgentResult:
        self.last_prompt = (
            f"{self.PROMPT}\n\n"
            f"SOLICITUD_USUARIO_START\n{user_request}\nSOLICITUD_USUARIO_END"
        )

        start = time.perf_counter()
        answer = self.llm.generate(self.last_prompt)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

        return AgentResult(
            agent=self.name,
            status="success",
            answer=answer,
            metadata={"execution_time_ms": elapsed_ms},
        )


class CreditAgent(SpecialistAgent):
    """Especialista en créditos, préstamos, financiación y tasas."""

    name = "credit_agent"
    description = "Especialista en créditos y préstamos"
    capabilities = ["prestamo", "credito", "financiacion", "tasa"]
    PROMPT = (
        "Eres un agente especialista en creditos, prestamos, financiacion "
        "y tasas.\n\n"
        "Responde la solicitud del usuario con una respuesta breve y clara, "
        "orientada a creditos y prestamos."
    )


class InvestmentAgent(SpecialistAgent):
    """Especialista en inversiones, bonos, acciones y fondos."""

    name = "investment_agent"
    description = "Especialista en inversiones"
    capabilities = ["inversion", "bono", "accion", "fondo"]
    PROMPT = (
        "Eres un agente especialista en inversiones, bonos, acciones y "
        "fondos.\n\n"
        "Responde la solicitud del usuario con una respuesta breve y clara, "
        "orientada a inversiones."
    )


class CardAgent(SpecialistAgent):
    """Especialista en tarjetas: límites, consumos y bloqueos."""

    name = "card_agent"
    description = "Especialista en tarjetas"
    capabilities = ["tarjeta", "limite", "consumo", "bloqueo"]
    PROMPT = (
        "Eres un agente especialista en tarjetas: limites, consumos y "
        "bloqueos.\n\n"
        "Responde la solicitud del usuario con una respuesta breve y clara, "
        "orientada a tarjetas."
    )
