# RAG simple con Pinecone

Misma idea que `../rag_simple.py`, pero usando **Pinecone** (base vectorial
administrada en la nube) en lugar de ChromaDB. Los embeddings y la generación
siguen siendo **locales** con Ollama (`nomic-embed-text` + `gemma3`).

La API key de Pinecone se lee de un archivo `.env` (nunca va en el código ni en el repo).

## Requisitos

1. **Cuenta gratis** en [pinecone.io](https://www.pinecone.io) y una **API key**
   (en https://app.pinecone.io → API Keys).
2. **Ollama** corriendo con los modelos:
   ```powershell
   ollama pull gemma3
   ollama pull nomic-embed-text
   ```
3. Dependencias (con el `.venv` de la raíz activado):
   ```powershell
   pip install -r requirements-pinecone.txt
   ```

## Configurar el token

```powershell
copy .env.example .env
```

Editá `.env` y completá `PINECONE_API_KEY=pcsk_...` con tu key real. El `.env`
**no se sube al repo** (está en `.gitignore`).

## Correr

```powershell
python rag_pinecone.py "¿Que es una base de datos vectorial?"

# o en modo interactivo
python rag_pinecone.py
```

## Qué hace

1. Se conecta a Pinecone con la API key del `.env`.
2. Crea el índice (si no existe) con la dimensión del embedding detectada
   automáticamente (nomic-embed-text = 768), métrica coseno, serverless.
3. Embebe el corpus de ejemplo y hace `upsert` de los vectores.
4. Para cada pregunta: embebe la consulta, busca los `top_k` más similares en
   Pinecone y le pasa ese contexto a Gemma para generar la respuesta.

## Diferencias con `rag_simple.py` (ChromaDB)

| | `rag_simple.py` (Chroma) | `rag_pinecone.py` (Pinecone) |
|---|---|---|
| Vector store | Local (embebido, en disco/memoria) | En la nube (administrado) |
| Setup | Cero config | Requiere API key |
| Escala | Prototipos | Producción / grandes volúmenes |
| Persistencia | Carpeta `chroma_db/` | Índice en tu cuenta de Pinecone |

## Notas

- El corpus se re-indexa (upsert) en cada corrida; como los `id` son fijos, se
  sobreescriben en vez de duplicarse.
- Se usa un `namespace` (`clase`) para aislar estos datos dentro del índice.
- El free tier de Pinecone alcanza de sobra para este ejemplo.
