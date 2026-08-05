# RAG sobre una carpeta de PDFs

Lee todos los PDFs de una carpeta, los indexa en una base vectorial local
(**ChromaDB** persistente) con embeddings de **Ollama**, y permite consultarlos
en lenguaje natural. Las respuestas citan el archivo y la página de origen.

## Requisitos

- **Ollama** con:
  ```powershell
  ollama pull gemma3
  ollama pull nomic-embed-text
  ```
- Dependencias:
  ```powershell
  pip install -r requirements-pdf.txt
  ```

## Uso

1. Poné tus PDFs en la carpeta `docs/` (o usá `--docs <ruta>`).
2. Corré:
   ```powershell
   # indexa (incremental) y responde una pregunta
   python rag_pdf.py "¿De que trata el informe?"

   # modo interactivo
   python rag_pdf.py

   # usar otra carpeta de PDFs
   python rag_pdf.py --docs C:\Users\yo\Documentos\papers "¿Que dice sobre X?"

   # reconstruir el indice desde cero
   python rag_pdf.py --reindex
   ```

## Cómo funciona

1. **Extracción**: `pypdf` lee el texto de cada página.
2. **Chunking**: cada página se parte en fragmentos de ~800 caracteres con
   solapamiento (para no cortar ideas).
3. **Indexado**: cada chunk se embebe (`nomic-embed-text`) y se guarda en ChromaDB
   con metadata (`source`, `page`). El índice persiste en `chroma_pdf/`.
4. **Consulta**: la pregunta se embebe, se recuperan los `top_k` chunks más
   similares y `gemma3` genera la respuesta citando las fuentes.

## Variante con LangChain + LangSmith (observabilidad)

`rag_pdf_langsmith.py` hace lo mismo pero construido con **LangChain** y con
**tracing en LangSmith**: cada consulta se registra en https://smith.langchain.com
(árbol retriever → prompt → LLM, latencias, tokens). Comparte la carpeta `docs/`
pero usa su propio índice (`chroma_pdf_lc/`).

```powershell
pip install -r requirements-pdf-langsmith.txt

# (opcional) para ver las trazas: token de LangSmith
copy .env.example .env
#  editá .env y completá LANGSMITH_API_KEY

python rag_pdf_langsmith.py "¿De que trata el documento?"
python rag_pdf_langsmith.py --reindex "..."   # reconstruir indice
```

Sin `LANGSMITH_API_KEY` el RAG funciona igual, solo que no registra trazas.

## Notas

- **Indexado incremental** (en `rag_pdf.py`): si un PDF ya fue indexado, se saltea.
  Agregá PDFs nuevos y volvé a correr; solo se indexan los nuevos. Usá `--reindex`
  para reconstruir todo. (La variante LangSmith indexa si el índice está vacío o
  con `--reindex`.)
- **PDFs escaneados** (imágenes sin texto) no tienen texto extraíble; requerirían
  OCR (no incluido en este ejemplo).
- La carpeta `chroma_pdf/` (índice) y los `docs/*.pdf` **no se suben al repo**
  (están en `.gitignore`).
