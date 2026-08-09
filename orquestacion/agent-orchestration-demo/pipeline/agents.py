"""
Agentes del pipeline secuencial.

Cada agente:

1. Tiene un PROMPT fijo (P1, P2 o P3) que describe su única responsabilidad.
2. Recibe datos concretos (la solicitud del usuario y/o la salida del
   agente anterior).
3. Arma un prompt completo (instrucciones + datos) usando marcadores de
   texto (SOLICITUD_USUARIO_START/END, TITULOS_START/END, etc).
4. Llama a `self.llm.generate(prompt)`.
5. Parsea la respuesta del LLM y la devuelve como una estructura simple
   (lista, lista de dicts o string) para que el siguiente agente la use.

`self.last_prompt` guarda el último prompt enviado, para que el
orquestador pueda mostrarlo en consola sin tener que reconstruirlo.
"""


class TitleAgent:
    """A1 - Genera los títulos principales del documento."""

    PROMPT = (
        "A partir de la solicitud del usuario, genera una lista de\n"
        "títulos para estructurar un documento.\n\n"
        "No desarrolles todavía el contenido.\n"
        "Devuelve solamente los títulos."
    )

    def __init__(self, llm):
        self.llm = llm
        self.last_prompt = ""

    def run(self, user_request: str) -> list[str]:
        self.last_prompt = (
            f"{self.PROMPT}\n\n"
            f"SOLICITUD_USUARIO_START\n{user_request}\nSOLICITUD_USUARIO_END"
        )
        raw_output = self.llm.generate(self.last_prompt)
        titles = [line.strip() for line in raw_output.splitlines() if line.strip()]
        return titles


class ContentAgent:
    """A2 - Genera contenido breve para cada título."""

    PROMPT = (
        "Utilizando los títulos proporcionados, genera contenido\n"
        "breve para cada sección.\n\n"
        "Mantén coherencia entre las secciones."
    )

    def __init__(self, llm):
        self.llm = llm
        self.last_prompt = ""

    def run(self, user_request: str, titles: list[str]) -> list[dict]:
        titles_block = "\n".join(f"- {title}" for title in titles)
        self.last_prompt = (
            f"{self.PROMPT}\n\n"
            f"SOLICITUD_USUARIO_START\n{user_request}\nSOLICITUD_USUARIO_END\n\n"
            f"TITULOS_START\n{titles_block}\nTITULOS_END"
        )
        raw_output = self.llm.generate(self.last_prompt)

        sections = []
        for block in raw_output.split("\n---\n"):
            block = block.strip()
            if not block:
                continue
            title_line, content_line = block.split("\n", 1)
            sections.append(
                {
                    "title": title_line.replace("TITULO:", "").strip(),
                    "content": content_line.replace("CONTENIDO:", "").strip(),
                }
            )
        return sections


class EditorAgent:
    """A3 - Revisa títulos y contenido y produce el documento final."""

    PROMPT = (
        "Revisa los títulos y el contenido generado.\n\n"
        "Mejora los títulos para que sean claros,\n"
        "consistentes y representen correctamente\n"
        "el contenido de cada sección."
    )

    def __init__(self, llm):
        self.llm = llm
        self.last_prompt = ""

    def run(self, user_request: str, titles: list[str], content: list[dict]) -> str:
        titles_block = "\n".join(f"- {title}" for title in titles)
        content_block = "\n---\n".join(
            f"TITULO: {item['title']}\nCONTENIDO: {item['content']}"
            for item in content
        )
        self.last_prompt = (
            f"{self.PROMPT}\n\n"
            f"TITULOS_START\n{titles_block}\nTITULOS_END\n\n"
            f"CONTENIDO_START\n{content_block}\nCONTENIDO_END"
        )
        final_document = self.llm.generate(self.last_prompt)
        return final_document
