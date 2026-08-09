"""
MockLLM
=======
Simula el comportamiento de un modelo de lenguaje (LLM) sin necesidad de
internet, API keys ni infraestructura externa.

Existe únicamente para que este proyecto pueda ejecutarse de forma 100%
local y determinística. El resto del código (agentes, orquestador, router,
aggregator) NO sabe ni le importa que el LLM sea "mock" o real: todos se
comunican con él a través de una única interfaz:

    MockLLM().generate(prompt: str) -> str

Cada agente arma un prompt de texto con marcadores tipo:

    SOLICITUD_USUARIO_START
    ...
    SOLICITUD_USUARIO_END

y MockLLM simplemente lee esos marcadores para simular una respuesta
coherente con lo que un LLM real generaría a partir de ese mismo prompt.

============================================================
CÓMO REEMPLAZAR MockLLM POR UN LLM REAL (por ejemplo Claude)
============================================================

1. Crear una clase con la MISMA interfaz, por ejemplo:

    class ClaudeLLM:
        def __init__(self, client, model="claude-sonnet-5"):
            self.client = client
            self.model = model

        def generate(self, prompt: str) -> str:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

2. En los agentes (pipeline/agents.py, router/agents.py), inyectar
   ClaudeLLM(...) en vez de MockLLM() al construirlos.

3. Nada más cambia. Agentes, orquestador, router y aggregator solo
   dependen de `generate(prompt) -> str`, nunca de la implementación
   concreta del modelo. Ese es el punto clave del diseño.
"""


_ACCENTS = str.maketrans("áéíóúÁÉÍÓÚñÑ", "aeiouAEIOUnN")


def _normalize(text: str) -> str:
    """Minúsculas + sin acentos, solo para detectar palabras clave."""
    return text.translate(_ACCENTS).lower()


def extract_between(text: str, start_marker: str, end_marker: str) -> str:
    """Extrae el texto ubicado entre dos marcadores literales.

    Se usa para leer, dentro del prompt completo, únicamente la sección
    que a cada método de MockLLM le interesa (p. ej. solo la solicitud
    del usuario, o solo la lista de títulos).
    """
    if start_marker not in text or end_marker not in text:
        return ""
    section = text.split(start_marker, 1)[1]
    section = section.split(end_marker, 1)[0]
    return section.strip()


class MockLLM:
    """LLM simulado con respuestas determinísticas basadas en el prompt."""

    def generate(self, prompt: str) -> str:
        # El "dispatch" se hace mirando qué instrucción (P1/P2/P3 o el
        # prompt de un agente especialista) contiene el prompt recibido.
        # Un LLM real no necesita esto: aquí simplemente reemplaza la
        # inferencia del modelo por reglas fijas y legibles.
        normalized = _normalize(prompt)

        if "lista de" in normalized and "titulos" in normalized:
            return self._generate_titles(prompt)
        if "genera contenido" in normalized:
            return self._generate_content(prompt)
        if "revisa los titulos y el contenido" in normalized:
            return self._generate_edited_document(prompt)
        if "especialista en creditos" in normalized:
            return self._generate_specialist_answer(prompt, "credit_agent")
        if "especialista en inversiones" in normalized:
            return self._generate_specialist_answer(prompt, "investment_agent")
        if "especialista en tarjetas" in normalized:
            return self._generate_specialist_answer(prompt, "card_agent")

        return "MockLLM: no se reconoce el tipo de prompt recibido."

    # ------------------------------------------------------------------
    # PIPELINE: A1 - generación de títulos
    # ------------------------------------------------------------------
    def _generate_titles(self, prompt: str) -> str:
        request = extract_between(
            prompt, "SOLICITUD_USUARIO_START", "SOLICITUD_USUARIO_END"
        )
        topic = self._extract_topic(request)

        titles = [
            f"Introduccion a {topic}",
            f"Casos de uso de {topic}",
            f"Beneficios de {topic}",
            f"Riesgos de {topic}",
            f"Gobernanza de {topic}",
        ]
        return "\n".join(titles)

    # ------------------------------------------------------------------
    # PIPELINE: A2 - generación de contenido
    # ------------------------------------------------------------------
    def _generate_content(self, prompt: str) -> str:
        request = extract_between(
            prompt, "SOLICITUD_USUARIO_START", "SOLICITUD_USUARIO_END"
        )
        titles_block = extract_between(prompt, "TITULOS_START", "TITULOS_END")
        titles = [
            line.lstrip("- ").strip()
            for line in titles_block.splitlines()
            if line.strip()
        ]

        blocks = []
        for title in titles:
            content = (
                f"Contenido breve sobre '{title}', desarrollado en el "
                f'contexto de la solicitud: "{request}". Esta seccion '
                f"mantiene coherencia con el resto del documento."
            )
            blocks.append(f"TITULO: {title}\nCONTENIDO: {content}")

        return "\n---\n".join(blocks)

    # ------------------------------------------------------------------
    # PIPELINE: A3 - edición final
    # ------------------------------------------------------------------
    def _generate_edited_document(self, prompt: str) -> str:
        titles_block = extract_between(prompt, "TITULOS_START", "TITULOS_END")
        content_block = extract_between(prompt, "CONTENIDO_START", "CONTENIDO_END")

        titles = [
            line.lstrip("- ").strip()
            for line in titles_block.splitlines()
            if line.strip()
        ]
        sections = content_block.split("\n---\n") if content_block else []

        lines = ["DOCUMENTO FINAL", "=" * 40, ""]
        for i, (title, section) in enumerate(zip(titles, sections), start=1):
            content = section.split("CONTENIDO:", 1)[-1].strip()
            improved_title = f"{i}. {title.rstrip('.')}"
            lines.append(improved_title)
            lines.append("-" * len(improved_title))
            lines.append(content)
            lines.append("")

        return "\n".join(lines).strip()

    # ------------------------------------------------------------------
    # ROUTER: respuesta de un agente especialista
    # ------------------------------------------------------------------
    def _generate_specialist_answer(self, prompt: str, agent_name: str) -> str:
        request = extract_between(
            prompt, "SOLICITUD_USUARIO_START", "SOLICITUD_USUARIO_END"
        )
        return f'[{agent_name}] Respuesta simulada para: "{request}"'

    # ------------------------------------------------------------------
    # Utilidades internas
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_topic(request: str) -> str:
        """Extrae un 'tema' simple de la solicitud del usuario."""
        request = request.strip().rstrip(".")
        if " sobre " in request:
            return request.split(" sobre ", 1)[1].strip()
        return request
