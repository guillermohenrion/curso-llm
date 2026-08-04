# RAG

**Summary**: RAG (Retrieval-Augmented Generation) combina la recuperacion de documentos con la generacion de texto para dar respuestas fundamentadas.
**Tags**: #rag #recuperacion #generacion #llm

---

## Content

RAG es una tecnica que primero **recupera** documentos relevantes de una base de
conocimiento y luego usa un **LLM** para **generar** la respuesta apoyandose en esos
documentos como contexto.

El pipeline tipico tiene tres etapas:

1. Ingesta y *chunking* de los documentos.
2. Indexado (por ejemplo en una base vectorial).
3. En consulta: recuperacion de los fragmentos mas relevantes + generacion.

RAG **mejora** a un LLM porque le aporta conocimiento externo y actualizado, y
reduce las alucinaciones al obligarlo a responder desde el contexto recuperado.

## Related Notes

- [[Bases vectoriales]]
- [[LangChain]]
- [[Ollama]]
