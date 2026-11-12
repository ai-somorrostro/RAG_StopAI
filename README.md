# RAG_StopAI

RAG sobre las leyes y normativa sobre la Inteligencia Artificial en España.

## Estructura

- `data/`: documentos en markdown usados por el RAG.
- `docs/`: PDFs originales.
- `Notebook.ipynb`: pipeline de preparación del índice (chunking, embeddings y FAISS).
- `rag.py`: lógica reutilizable de carga, indexación, búsqueda y generación.
- `app.py`: interfaz web con Streamlit para preguntar al RAG.

## Uso

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

La primera ejecución descarga el modelo de embeddings y construye el índice en
`rag_store/` a partir de los documentos de `data/`.

## Respuestas con LLM (opcional)

Por defecto la interfaz muestra los fragmentos recuperados (modo extractivo).
Para obtener respuestas redactadas, activa la opción de Ollama en la barra
lateral y ejecuta un modelo local:

```bash
ollama serve
ollama pull llama3.1
```
