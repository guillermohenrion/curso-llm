# Ollama

**Summary**: Ollama permite correr LLMs y modelos de embeddings de forma local, sin depender de una API en la nube.
**Tags**: #ollama #local #gemma #embeddings

---

## Content

**Ollama** es una herramienta para correr modelos de lenguaje grandes en tu
propia maquina. Se descargan con `ollama pull <modelo>` y se consultan por una
API local en `http://localhost:11434`.

Modelos usados en el curso:

- **gemma3**: modelo generador (chat).
- **nomic-embed-text**: modelo de embeddings (genera vectores de 768 dimensiones).

Al ser local, no manda datos a la nube y no tiene costo por token.

## Related Notes

- [[RAG]]
- [[Bases vectoriales]]
- [[LangChain]]
