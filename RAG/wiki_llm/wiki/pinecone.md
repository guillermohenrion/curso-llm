# Pinecone

**Summary**: Pinecone es una base de datos vectorial administrada en la nube; se crea un indice, se hace upsert de vectores y se consultan los mas similares.
**Tags**: #pinecone #bases-vectoriales #nube #indice

---

## Content

**Pinecone** es una base vectorial *serverless* administrada. No hay que
mantener infraestructura: se crea un **indice** (con una dimension y una metrica,
por ejemplo coseno), se hace **upsert** de vectores con metadata, y se consultan
los `top_k` mas similares.

Se accede con una **API key** (que conviene leer de un archivo `.env`, nunca
hardcodearla). Es una alternativa en la nube a [[Bases vectoriales]] locales como Chroma.

## Related Notes

- [[Bases vectoriales]]
- [[RAG]]
