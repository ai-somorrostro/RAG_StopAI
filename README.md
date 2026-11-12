# RAG_StopAI
RAG sobre las leyes y normativa sobre la Inteligencia Artificial en España

## Consulta interactiva por terminal

El proyecto incluye un chat RAG que recupera fragmentos de los documentos
indexados y los envía a un modelo local servido por [Ollama](https://ollama.com/).
Las preguntas y los documentos no se envían a una API remota.

### Preparación

1. Instala Ollama y descarga un modelo:

   ```bash
   ollama pull qwen2.5:3b
   ```

   Ollama debe estar ejecutándose. En Linux, si no se inició como servicio,
   arráncalo en otra terminal con `ollama serve`.

2. Desde la carpeta del proyecto, crea un entorno e instala las dependencias:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   ```

3. Si no existen los archivos de `rag_store/`, abre `Notebook.ipynb` y ejecuta
   todas sus celdas para generarlos. El modelo de embeddings se descargará la
   primera vez que se ejecute.

4. Inicia el chat:

   ```bash
   python chat.py
   ```

Escribe una consulta y el programa mostrará la respuesta y los fragmentos
recuperados como fuentes. Puedes hacer consultas sucesivas; escribe `salir`,
`exit` o `quit` para terminar. Cada consulta se responde usando sus propios
fragmentos recuperados.

El modelo predeterminado es `qwen2.5:3b`, elegido para funcionar en equipos con
memoria limitada. Puedes elegir otro que tengas
descargado en Ollama mediante `OLLAMA_MODEL=nombre python chat.py`, o usar las
opciones `--model`, `--ollama-url` y `--top-k`. Verifica siempre la versión de
la legislación en la fuente oficial antes de usar una respuesta con fines
jurídicos.

### Pruebas

```bash
python -m unittest discover -s tests
```
