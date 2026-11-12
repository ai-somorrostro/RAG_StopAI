import json
import re
from pathlib import Path

import faiss
import requests
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
OUTPUT_DIR = Path("rag_store")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".html", ".htm"}
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.1"


def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


def read_pdf(path):
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def read_html(path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return text


def read_text(path):
    return path.read_text(encoding="utf-8", errors="ignore")


def load_documents(data_dir=DATA_DIR):
    if not data_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta '{data_dir}'.")

    documents = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        if path.suffix.lower() == ".pdf":
            content = read_pdf(path)
        elif path.suffix.lower() in {".html", ".htm"}:
            content = read_html(path)
        else:
            content = read_text(path)

        documents.append({"source": str(path), "content": content})

    return documents


def clean_text(text):
    text = text.replace("\u00ad", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], ""

    for sentence in sentences:
        if not sentence:
            continue

        if len(sentence) > chunk_size:
            step = max(chunk_size - overlap, 1)
            for i in range(0, len(sentence), step):
                chunks.append(sentence[i:i + chunk_size])
            current = ""
            continue

        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= chunk_size:
            current = f"{current} {sentence}"
        else:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail} {sentence}".strip()

    if current:
        chunks.append(current)

    return chunks


def build_index(data_dir=DATA_DIR, output_dir=OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)

    documents = load_documents(data_dir)
    if not documents:
        raise ValueError(f"No hay documentos en '{data_dir}'.")

    chunks = []
    for doc in documents:
        cleaned = clean_text(doc["content"])
        for chunk_id, chunk in enumerate(chunk_text(cleaned)):
            chunks.append(
                {
                    "source": doc["source"],
                    "chunk_id": chunk_id,
                    "text": chunk,
                }
            )

    if not chunks:
        raise ValueError("No se han generado chunks con texto extraíble.")

    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(
        [chunk["text"] for chunk in chunks],
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(output_dir / "index.faiss"))

    (output_dir / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metadata = {
        "embedding_model": EMBEDDING_MODEL,
        "num_documents": len(documents),
        "num_chunks": len(chunks),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "dimension": int(embeddings.shape[1]),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return metadata


def load_index(output_dir=OUTPUT_DIR):
    index = faiss.read_index(str(output_dir / "index.faiss"))
    chunks = json.loads((output_dir / "chunks.json").read_text(encoding="utf-8"))
    metadata_path = output_dir / "metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    return index, chunks, metadata


def search(model, index, chunks, query, k=5):
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")
    scores, indices = index.search(query_embedding, k)
    return [
        {"score": float(score), "chunk": chunks[index]}
        for score, index in zip(scores[0], indices[0])
        if index != -1
    ]


def format_context(hits):
    blocks = []
    for position, hit in enumerate(hits, start=1):
        chunk = hit["chunk"]
        source = Path(chunk["source"]).name
        blocks.append(f"[{position}] ({source})\n{chunk['text']}")
    return "\n\n".join(blocks)


def ask_ollama(question, context, model=DEFAULT_OLLAMA_MODEL, url=DEFAULT_OLLAMA_URL, timeout=120):
    prompt = (
        "Eres un asistente experto en normativa de inteligencia artificial en España "
        "y la Unión Europea. Responde en español usando únicamente el contexto "
        "proporcionado. Si la respuesta no está en el contexto, di que no lo sabes. "
        "Cita las fuentes con el número entre corchetes, por ejemplo [1].\n\n"
        f"Contexto:\n{context}\n\n"
        f"Pregunta: {question}\n\n"
        "Respuesta:"
    )
    response = requests.post(
        f"{url.rstrip('/')}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


def extractive_answer(hits):
    if not hits:
        return "No he encontrado fragmentos relevantes para esa pregunta."
    lines = ["Estos son los fragmentos más relevantes de la normativa:\n"]
    for position, hit in enumerate(hits, start=1):
        source = Path(hit["chunk"]["source"]).name
        text = hit["chunk"]["text"]
        if len(text) > 600:
            text = text[:600].rsplit(" ", 1)[0] + "…"
        lines.append(f"**[{position}] {source}**\n\n{text}\n")
    return "\n".join(lines)
