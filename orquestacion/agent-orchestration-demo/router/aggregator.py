"""
Aggregator
==========
Combina los resultados de uno o varios agentes especializados en una
única FinalResponse.

En esta primera versión NO se usa un LLM para agregar: simplemente se
concatenan las respuestas de cada agente. Es la forma más simple posible
de "cerrar" el flujo Router -> Agentes -> Respuesta final.
"""

from common.models import AgentResult, FinalResponse


class Aggregator:
    """Combina una lista de AgentResult en una FinalResponse."""

    def aggregate(self, user_request: str, results: list[AgentResult]) -> FinalResponse:
        # FUTURE:
        # Replace this aggregation logic with an LLM
        # that synthesizes the individual agent responses.
        successful = [r for r in results if r.status == "success"]

        if len(successful) == 1:
            combined_answer = successful[0].answer
        else:
            parts = [f"[{r.agent}]\n{r.answer}" for r in successful]
            combined_answer = "\n\n".join(parts)

        return FinalResponse(
            query=user_request,
            agents_used=[r.agent for r in results],
            answer=combined_answer,
            details=results,
        )
