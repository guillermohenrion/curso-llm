# Bases vectoriales

**Summary**: Una base de datos vectorial almacena embeddings y busca los vectores mas similares a una consulta.
**Tags**: #bases-vectoriales #embeddings #chroma #pinecone #similitud

---

## Content

Una **base de datos vectorial** guarda *embeddings* (vectores numericos que
representan el significado de un texto) y permite buscar los vectores mas
parecidos a una consulta usando metricas como la **distancia coseno**.

Opciones comunes:

- **Chroma**: embebida, corre local (en memoria o en disco). Ideal para prototipos.
- **Pinecone**: administrada, corre en la nube. Ideal para produccion y grandes volumenes.

Son la pieza que hace posible la etapa de *recuperacion* en un sistema [[RAG]].

## Related Notes

- [[RAG]]
- [[Pinecone]]
- [[Ollama]]
