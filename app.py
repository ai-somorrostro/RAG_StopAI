from pathlib import Path

import streamlit as st

import rag

st.set_page_config(
    page_title="Asistente de normativa de IA",
    page_icon="⚖️",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_model():
    return rag.load_embedding_model()


@st.cache_resource(show_spinner=False)
def get_store():
    return rag.load_index()


def ensure_index():
    index_path = rag.OUTPUT_DIR / "index.faiss"
    if index_path.exists():
        return
    with st.spinner("Construyendo el índice vectorial por primera vez…"):
        rag.build_index()
    st.toast("Índice construido correctamente.")


def render_sources(hits):
    if not hits:
        return
    with st.expander(f"Fuentes consultadas ({len(hits)})"):
        for position, hit in enumerate(hits, start=1):
            chunk = hit["chunk"]
            st.markdown(
                f"**[{position}] {Path(chunk['source']).name}** · "
                f"chunk {chunk['chunk_id']} · similitud {hit['score']:.3f}"
            )
            text = chunk["text"]
            st.caption(text if len(text) <= 800 else text[:800].rsplit(" ", 1)[0] + "…")
            st.divider()


st.title("⚖️ Asistente RAG de normativa de IA")
st.caption(
    "Pregunta sobre la normativa española y europea de inteligencia artificial. "
    "Las respuestas se basan en los documentos de `data/`."
)

ensure_index()

try:
    model = get_model()
    index, chunks, metadata = get_store()
except Exception as error:
    st.error(f"No se pudo cargar el índice: {error}")
    st.stop()

with st.sidebar:
    st.header("Configuración")

    if metadata:
        st.metric("Fragmentos indexados", metadata.get("num_chunks", len(chunks)))
        st.metric("Documentos", metadata.get("num_documents", "—"))
        st.caption(f"Modelo de embeddings: `{metadata.get('embedding_model', rag.EMBEDDING_MODEL)}`")

    st.divider()
    k = st.slider("Fragmentos a recuperar", min_value=1, max_value=8, value=4)

    st.divider()
    use_llm = st.toggle("Generar respuesta con LLM local (Ollama)", value=False)
    ollama_url = rag.DEFAULT_OLLAMA_URL
    ollama_model = rag.DEFAULT_OLLAMA_MODEL
    if use_llm:
        ollama_url = st.text_input("URL de Ollama", value=rag.DEFAULT_OLLAMA_URL)
        ollama_model = st.text_input("Modelo", value=rag.DEFAULT_OLLAMA_MODEL)
        st.caption("Si Ollama no está disponible se mostrarán los fragmentos recuperados.")

    st.divider()
    if st.button("Reconstruir índice", use_container_width=True):
        with st.spinner("Reconstruyendo el índice…"):
            rag.build_index()
        get_store.clear()
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))

if not st.session_state.messages:
    st.info(
        "Ejemplos: «¿Qué obligaciones establece la normativa de IA en España?» · "
        "«¿Qué sistemas de IA se consideran de alto riesgo?» · "
        "«¿Qué dice el Reglamento Europeo de IA sobre transparencia?»"
    )

question = st.chat_input("Escribe tu pregunta sobre la normativa…")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Buscando en la normativa…"):
            hits = rag.search(model, index, chunks, question, k=k)
            context = rag.format_context(hits)

            answer = None
            if use_llm:
                try:
                    answer = rag.ask_ollama(
                        question,
                        context,
                        model=ollama_model,
                        url=ollama_url,
                    )
                except Exception as error:
                    st.warning(
                        f"No se pudo contactar con Ollama ({error}). "
                        "Se muestran los fragmentos recuperados."
                    )

            if not answer:
                answer = rag.extractive_answer(hits)

            st.markdown(answer)
            render_sources(hits)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": hits}
    )
