import json
import sys
import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

import rag


class FakeVector:
    shape = (1, 2)

    def reshape(self, *_shape):
        return self


class FakeNumpy:
    @staticmethod
    def asarray(_values, dtype=None):
        return FakeVector()


class FakeEmbedder:
    def encode(self, texts, **_kwargs):
        self.encoded_texts = texts
        return [[0.1, 0.2]]


class FakeIndex:
    ntotal = 2
    d = 2

    def search(self, _query_vector, k):
        self.searched_k = k
        return [[0.91, 0.82][:k]], [[1, 0][:k]]


class RAGTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {
                "source": "data/Reglamento-Europeo-IA.md",
                "legal_reference": "Reglamento (UE) 2024/1689",
                "article": "9",
                "section": "Artículo 9",
                "source_version": "10/11/2026",
                "chunk_id": 0,
                "text": "El sistema de gestión de riesgos.",
            },
            {
                "source": "data/LOPDGDD.md",
                "legal_reference": "BOE-A-2018-16673",
                "article": None,
                "section": "Preámbulo",
                "chunk_id": 0,
                "text": "Texto de referencia.",
            },
        ]
        self.metadata = {"embedding_model": "test-embedder", "dimension": 2}
        self.store = rag.RAGStore(
            FakeIndex(), self.chunks, FakeEmbedder(), FakeNumpy(), self.metadata
        )

    def test_search_encodes_query_and_caps_k_to_index_size(self):
        results = self.store.search("gestión de riesgos", k=5)

        self.assertEqual(self.store.embedder.encoded_texts, ["gestión de riesgos"])
        self.assertEqual(self.store.index.searched_k, 2)
        self.assertEqual([chunk for _, chunk in results], [self.chunks[1], self.chunks[0]])

    def test_search_rejects_empty_question_invalid_k_and_incompatible_embeddings(self):
        with self.assertRaises(ValueError):
            self.store.search("  ")
        with self.assertRaises(ValueError):
            self.store.search("consulta", k=0)
        self.store.index.d = 3
        with self.assertRaisesRegex(RuntimeError, "dimensión"):
            self.store.search("consulta")

    def test_prompt_includes_question_sources_and_grounding_instructions(self):
        messages = rag.build_messages("¿Qué exige el artículo 9?", [(0.9, self.chunks[0])])

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("únicamente", messages[0]["content"])
        self.assertIn("¿Qué exige el artículo 9?", messages[1]["content"])
        self.assertIn("[1] Reglamento (UE) 2024/1689 — artículo 9", messages[1]["content"])
        self.assertIn("10/11/2026", messages[1]["content"])
        self.assertIn("El sistema de gestión de riesgos.", messages[1]["content"])

    def test_citation_uses_section_when_chunk_has_no_article(self):
        citation = rag.format_citation(self.chunks[1])

        self.assertIn("BOE-A-2018-16673", citation)
        self.assertIn("Preámbulo", citation)
        self.assertIn("data/LOPDGDD.md", citation)

    def test_ollama_request_uses_non_streaming_chat_endpoint(self):
        messages = [{"role": "user", "content": "Pregunta"}]
        with patch(
            "rag.urlopen",
            return_value=BytesIO(b'{"message": {"content": " Respuesta [1]. "}}'),
        ) as urlopen:
            answer = rag.ask_ollama(messages, "test-model", "http://localhost:11434/")

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(answer, "Respuesta [1].")
        self.assertEqual(request.full_url, "http://localhost:11434/api/chat")
        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(payload["messages"], messages)
        self.assertFalse(payload["stream"])

    def test_ollama_rejects_malformed_empty_and_error_responses(self):
        for body in (
            b"not-json",
            b"[]",
            b'{"message": {"content": " "}}',
            b'{"message": {}}',
            b'{"error": "modelo no disponible"}',
        ):
            with self.subTest(body=body), patch("rag.urlopen", return_value=BytesIO(body)):
                with self.assertRaises(RuntimeError):
                    rag.ask_ollama([], "test-model", rag.DEFAULT_OLLAMA_URL)

    def test_ollama_reports_connection_timeout_and_http_failures(self):
        errors = [
            (URLError("sin conexión"), "No se pudo conectar"),
            (TimeoutError(), "no respondió"),
            (
                HTTPError(
                    "http://localhost:11434/api/chat",
                    404,
                    "Not Found",
                    None,
                    BytesIO(b"modelo no instalado"),
                ),
                "HTTP 404",
            ),
        ]
        for error, expected in errors:
            with self.subTest(error=error), patch("rag.urlopen", side_effect=error):
                with self.assertRaisesRegex(RuntimeError, expected):
                    rag.ask_ollama([], "test-model", rag.DEFAULT_OLLAMA_URL)

    def test_shared_answer_pipeline_passes_retrieved_citations_and_configuration(self):
        with patch("rag.ask_ollama", return_value="respuesta") as ask:
            answer, results = rag.answer_question(
                self.store, "pregunta", k=1, model="local-model", ollama_url="http://localhost:1234"
            )

        self.assertEqual(answer, "respuesta")
        self.assertEqual(
            ask.call_args.args,
            (rag.build_messages("pregunta", results), "local-model", "http://localhost:1234"),
        )
        self.assertEqual(self.store.index.searched_k, 1)
        self.assertEqual(len(results), 1)

    def test_retrieval_only_and_no_results_do_not_call_ollama(self):
        with patch("rag.ask_ollama") as ask:
            answer, results = rag.answer_question(self.store, "pregunta", use_llm=False)
            self.assertIn("fragmentos más relevantes", answer)
            self.assertIn(rag.format_citation(results[0][1]), answer)
            with patch.object(self.store, "search", return_value=[]):
                answer, results = rag.answer_question(self.store, "pregunta")
            self.assertIn("No se encontraron", answer)
            self.assertEqual(results, [])
        ask.assert_not_called()

    def test_generation_failure_does_not_fall_back_to_excerpts(self):
        with (
            patch("rag.ask_ollama", side_effect=RuntimeError("Ollama no disponible")),
            patch("rag.extractive_answer") as extract,
        ):
            with self.assertRaisesRegex(RuntimeError, "Ollama no disponible"):
                rag.answer_question(self.store, "pregunta")
        extract.assert_not_called()

    def test_missing_store_is_not_created_and_points_to_notebook(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "missing-store"
            with self.assertRaisesRegex(RuntimeError, "crear_rag.ipynb"):
                rag.load_rag_store(path)
            self.assertFalse(path.exists())

    def test_load_store_uses_metadata_model_and_preserves_artifacts(self):
        with TemporaryDirectory() as directory:
            path = self.write_store(directory)
            before = {file.name: file.read_bytes() for file in path.iterdir()}
            sentence_transformers = SimpleNamespace(
                SentenceTransformer=Mock(return_value=FakeEmbedder())
            )
            modules = {
                "faiss": SimpleNamespace(read_index=Mock(return_value=FakeIndex())),
                "numpy": FakeNumpy(),
                "sentence_transformers": sentence_transformers,
            }
            with patch.dict(sys.modules, modules):
                store = rag.load_rag_store(path)
            after = {file.name: file.read_bytes() for file in path.iterdir()}

        sentence_transformers.SentenceTransformer.assert_called_once_with("test-embedder")
        self.assertEqual(store.metadata, self.metadata)
        self.assertEqual(store.chunks, self.chunks)
        self.assertEqual(before, after)

    def test_load_store_rejects_corrupted_json_chunks_and_metadata(self):
        cases = [
            ("chunks.json", "{"),
            ("chunks.json", "[]"),
            ("chunks.json", '[{"text": "sin fuente"}]'),
            ("metadata.json", "{}"),
        ]
        for filename, content in cases:
            with self.subTest(filename=filename, content=content), TemporaryDirectory() as directory:
                path = self.write_store(directory)
                (path / filename).write_text(content, encoding="utf-8")
                with self.assertRaises(RuntimeError):
                    rag.load_rag_store(path)

    def test_load_store_rejects_vector_count_and_dimension_before_model_loading(self):
        for index in (
            SimpleNamespace(ntotal=1, d=2),
            SimpleNamespace(ntotal=2, d=3),
        ):
            with self.subTest(index=index), TemporaryDirectory() as directory:
                path = self.write_store(directory)
                constructor = Mock()
                modules = {
                    "faiss": SimpleNamespace(read_index=lambda _path: index),
                    "numpy": FakeNumpy(),
                    "sentence_transformers": SimpleNamespace(SentenceTransformer=constructor),
                }
                with patch.dict(sys.modules, modules):
                    with self.assertRaises(RuntimeError):
                        rag.load_rag_store(path)
                constructor.assert_not_called()

    def write_store(self, directory):
        path = Path(directory)
        (path / "index.faiss").write_bytes(b"fake-index")
        (path / "chunks.json").write_text(json.dumps(self.chunks), encoding="utf-8")
        (path / "metadata.json").write_text(json.dumps(self.metadata), encoding="utf-8")
        return path


if __name__ == "__main__":
    unittest.main()
