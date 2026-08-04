# LangChain

**Summary**: LangChain es un framework para construir aplicaciones con LLMs: prompts, cadenas, retrievers, memoria, agentes y tools.
**Tags**: #langchain #framework #agentes #tools #memoria

---

## Content

**LangChain** ofrece componentes reutilizables para armar apps con LLMs y un
lenguaje de composicion llamado **LCEL** (se encadenan piezas con el operador `|`).

Componentes clave:

- **Retrievers**: recuperan documentos (base de la etapa de recuperacion de [[RAG]]).
- **Memoria**: mantiene el historial de una conversacion.
- **Agentes**: deciden que **tool** usar en cada paso (patron ReAct).
- **Tools**: funciones que el agente puede invocar (buscar, calcular, etc.).

En el curso hay 5 ejemplos incrementales: basico, memoria, tools, skills y LangSmith.

## Related Notes

- [[RAG]]
- [[Ollama]]
