# RAG_StopAI
RAG sobre las leyes y normativa sobre la Inteligencia Artificial en España.

## Estructura y flujo

1. `crear_rag.ipynb` es el único punto de creación del RAG: carga los documentos
   Markdown de `data/`, limpia y divide el texto por secciones legales, genera
   embeddings y guarda el índice FAISS con metadatos de cita.
2. `rag_store/` contiene `index.faiss`, `chunks.json` y `metadata.json`. Son
   archivos generados localmente y excluidos de Git; cada entorno debe
   generarlos con el notebook o recibir una copia compatible.
3. `rag.py` es la conexión compartida: valida y carga el índice existente,
   recupera fragmentos, prepara las citas y consulta Ollama. No crea ni
   modifica el índice.
4. `app.py` (Streamlit) y `chat.py` (terminal) usan esa misma conexión.

El modelo de embeddings es
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Al consultar,
se carga el modelo indicado en los metadatos del índice para mantener la
compatibilidad.

El modelo generativo predeterminado en ambos clientes es **`qwen2.5:3b`**,
servido localmente por [Ollama](https://ollama.com/). Los clientes no descargan
ni arrancan el LLM. Con la URL predeterminada, las preguntas y los fragmentos se
envían únicamente al servidor local; no configures una URL remota si los datos
deben permanecer en el equipo.

## Preparación

**En un equipo sin recursos suficientes, no ejecutes el notebook, los clientes
ni los comandos de descarga o arranque de modelos.** Las pruebas descritas al
final usan dobles de prueba y no requieren modelos.

### 1. Entorno y dependencias

Desde la carpeta del proyecto, con Python 3.10 o posterior:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

En Linux/macOS, activa el entorno con `source .venv/bin/activate`.
Selecciona el mismo entorno como kernel en tu editor de notebooks.

### 2. Crear el índice

En un equipo adecuado, abre `crear_rag.ipynb` desde la carpeta del proyecto y
ejecuta sus celdas. La primera ejecución descargará el modelo de embeddings.
El notebook conserva las referencias legales, artículos y versiones de las
fuentes junto a los fragmentos.

Si falta el índice, los clientes indican que debes ejecutar el notebook; no
lo reconstruyen automáticamente. Tras regenerarlo, reinicia los clientes para
recargarlo (Streamlit mantiene el índice en caché).

### 3. Preparar Ollama

Solo en el equipo que ejecutará el LLM, instala Ollama y descarga el modelo:

```powershell
ollama pull qwen2.5:3b
```

Ollama debe estar ejecutándose. Si no se inició como servicio o aplicación,
arráncalo en otra terminal con `ollama serve`.

## Consulta por terminal

```powershell
python chat.py
```

El chat muestra la respuesta y los fragmentos usados como fuentes. Puedes
hacer consultas sucesivas; escribe `salir`, `exit` o `quit` para terminar.
Cada consulta utiliza sus propios fragmentos, sin historial enviado al modelo.

Puedes elegir otro modelo ya instalado o cambiar el servidor y la cantidad de
fragmentos:

```powershell
python chat.py --model qwen2.5:3b --ollama-url http://localhost:11434 --top-k 5
```

Ambos clientes aceptan también las variables de entorno `OLLAMA_MODEL` y
`OLLAMA_URL`. Por ejemplo, en PowerShell:

```powershell
$env:OLLAMA_MODEL = "qwen2.5:3b"
$env:OLLAMA_URL = "http://localhost:11434"
python chat.py
```

## Interfaz web

```powershell
python -m streamlit run app.py
```

La generación local con Ollama está activada por defecto. En la barra lateral
puedes configurar el modelo, la URL y el número de fragmentos, o desactivar
la generación para mostrar solo los fragmentos recuperados. **Este último modo
también ejecuta el modelo de embeddings.** Si Ollama falla, se muestra el error:
no se sustituye una respuesta generada por fragmentos de forma automática.

Las respuestas citan las fuentes y deben indicar cuándo el contexto no basta.
Verifica siempre la versión de la legislación en la fuente oficial antes de
usar una respuesta con fines jurídicos.

## Pruebas sin modelos

```powershell
python -m unittest discover -s tests -v
```

Las pruebas usan índices y embeddings simulados y peticiones Ollama
interceptadas: no descargan modelos, no ejecutan inferencia ni requieren un
servidor Ollama. Las pruebas de interfaz usan Streamlit, incluido en
`requirements.txt`. Estas comprobaciones no evalúan la calidad real de las
respuestas ni la recuperación con el modelo.
