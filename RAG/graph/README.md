# Graph RAG: RAG sobre un grafo de conocimiento

En vez de recuperar fragmentos de texto de una base vectorial, la base de
conocimiento es un **grafo**: nodos = conceptos, aristas = relaciones
(tripletas `sujeto -> relacion -> objeto`).

## Cómo funciona

1. **Match de entidades**: se comparan los embeddings (Ollama `nomic-embed-text`)
   de la pregunta con los de cada nodo, y se eligen los más similares.
2. **Vecindario**: se recuperan las tripletas a 1-2 saltos de esas entidades
   (el subgrafo relevante).
3. **Generación**: ese subgrafo, serializado como `A --relacion--> B`, es el
   contexto que recibe el LLM (`gemma3`).

La ventaja del grafo es que el contexto son **relaciones explícitas y conectadas**,
útil para preguntas que requieren enlazar varios hechos.

## Requisitos

- **Ollama** con:
  ```powershell
  ollama pull gemma3
  ollama pull nomic-embed-text
  ```
- Dependencias:
  ```powershell
  pip install -r requirements-graph.txt
  ```

## Correr

```powershell
python rag_grafos.py "¿Que diferencia hay entre Chroma y Pinecone?"
python rag_grafos.py            # interactivo
```

## Ampliar el grafo

Editá la lista `TRIPLETAS` en `rag_grafos.py` para agregar nodos y relaciones.
El grafo se arma con `networkx` en memoria en cada corrida.
