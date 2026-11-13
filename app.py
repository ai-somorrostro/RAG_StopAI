import os

import streamlit as st

import rag

st.set_page_config(
    page_title="Asistente de normativa de IA",
    page_icon="⚖️",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_store():
    return rag.load_rag_store()


def render_sources(hits):
    if not hits:
        return
    with st.expander(f"Fuentes consultadas ({len(hits)})"):
        for position, (score, chunk) in enumerate(hits, start=1):
            st.markdown(
                f"**[{position}] {rag.format_citation(chunk)}** · "
                f"chunk {chunk['chunk_id']} · similitud {score:.3f}"
            )
            text = chunk["text"]
            st.caption(text if len(text) <= 800 else text[:800].rsplit(" ", 1)[0] + "…")
            st.divider()


st.title("⚖️ Asistente RAG de normativa de IA")
st.caption(
    "Pregunta sobre la normativa española y europea de inteligencia artificial. "
    "Las respuestas se basan en los documentos de `data/`."
)

try:
    store = get_store()
except (RuntimeError, ValueError, OSError) as error:
    st.error(f"No se pudo cargar el índice: {error}")
    st.stop()

metadata = store.metadata

with st.sidebar:
    st.header("Configuración")

    if metadata:
        st.metric("Fragmentos indexados", metadata.get("num_chunks", len(store.chunks)))
        st.metric("Documentos", metadata.get("num_documents", "—"))
        st.caption(f"Modelo de embeddings: `{metadata['embedding_model']}`")

    st.divider()
    k = st.slider("Fragmentos a recuperar", min_value=1, max_value=8, value=4)

    st.divider()
    use_llm = st.toggle("Generar respuesta con LLM local (Ollama)", value=True)
    ollama_url = os.getenv("OLLAMA_URL", rag.DEFAULT_OLLAMA_URL)
    ollama_model = os.getenv("OLLAMA_MODEL", rag.DEFAULT_MODEL)
    if use_llm:
        ollama_url = st.text_input("URL de Ollama", value=ollama_url)
        ollama_model = st.text_input("Modelo", value=ollama_model)
    st.caption(
        "Desactiva la generación para consultar solo los fragmentos recuperados. "
        "Este modo también necesita el modelo de embeddings."
    )

    st.divider()
    st.caption(
        "El índice se crea únicamente con crear_rag.ipynb. "
        "Después de regenerarlo, reinicia la aplicación para recargarlo."
    )

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
            try:
                answer, hits = rag.answer_question(
                    store,
                    question,
                    k=k,
                    model=ollama_model,
                    ollama_url=ollama_url,
                    use_llm=use_llm,
                )
            except (RuntimeError, ValueError, OSError) as error:
                st.error(f"No se pudo responder: {error}")
                st.stop()

            st.markdown(answer)
            render_sources(hits)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": hits}
    )
