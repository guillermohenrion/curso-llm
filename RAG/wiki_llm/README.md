# Wiki-LLM: RAG sobre una wiki de markdown (patrón de Karpathy)

Implementa el patrón del **"LLM wiki"** que popularizó Andrej Karpathy: en vez de
una base vectorial con embeddings, la base de conocimiento es una **carpeta de
archivos markdown** estructurados que el LLM lee directamente. La "recuperación"
es a nivel de **archivo** (no de chunks vectoriales).

Referencia: [What Is Andrej Karpathy's LLM Wiki](https://www.mindstudio.ai/blog/andrej-karpathy-llm-wiki-knowledge-base-claude-code).
_Contenido reformulado; ver el artículo original para el detalle._

## Idea

- Cada nota `.md` tiene una estructura consistente: **título**, línea **`Summary`**,
  **`Tags`**, contenido y enlaces `[[wiki]]` a otras notas.
- El `Summary` + `Tags` funcionan como un "índice liviano": permiten decidir qué
  notas son relevantes **sin** leer el archivo completo.
- No hay embeddings ni base vectorial: solo archivos + un LLM que los lee.

## Dos modos de carga (como en el artículo)

- **Selectivo** (default): se hace un *shortlist* de las notas relevantes por
  coincidencia con `Summary`/`Tags` (una especie de `grep`/fuzzy) y se cargan solo
  esas. Escala mejor cuando la wiki crece.
- **Full** (`--full`): se concatenan **todas** las notas. Simple y muchas veces
  mejor para wikis chicas, porque el modelo conecta temas entre notas.

En ambos casos se antepone un **system prompt de grounding** (responder solo desde
la wiki) y el modelo **cita** de qué archivos salió la respuesta.

## Requisitos

- **Ollama** corriendo con el modelo de chat:
  ```powershell
  ollama pull gemma3
  ```
- Dependencias (mínimas):
  ```powershell
  pip install -r requirements-wiki.txt
  ```

## Correr

```powershell
# Modo selectivo (shortlist por Summary/Tags)
python rag_wiki.py "¿Que diferencia hay entre Chroma y Pinecone?"

# Modo full (carga toda la wiki)
python rag_wiki.py --full "Resumi todo lo que hay sobre RAG"

# Interactivo
python rag_wiki.py
```

## Estructura

```
wiki_llm/
├── rag_wiki.py              # lee la wiki, arma el contexto y consulta al LLM
├── requirements-wiki.txt    # solo 'ollama'
└── wiki/                    # la base de conocimiento (markdown)
    ├── rag.md
    ├── bases_vectoriales.md
    ├── ollama.md
    ├── langchain.md
    └── pinecone.md
```

## Cómo crece la wiki

Agregá más notas `.md` en `wiki/` siguiendo el mismo template (título, `Summary`,
`Tags`, contenido, `Related Notes`). No hay que reindexar nada: el script las toma
en la próxima corrida. Ese es el atractivo del patrón — la base de conocimiento son
archivos de texto que vos (o el propio LLM) pueden editar.

## Diferencia con los otros RAG del curso

| Enfoque | Recuperación | Índice |
|---|---|---|
| `rag_simple.py` / `pinecone` | chunks por similitud de embeddings | base vectorial |
| `graph/` | vecindario en un grafo de conocimiento | grafo |
| **`wiki_llm/`** | **archivos markdown (shortlist por summary/tags)** | **el filesystem** |
