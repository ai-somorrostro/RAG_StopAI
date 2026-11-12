import json
from contextlib import redirect_stdout
from io import StringIO
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from chat import RAGStore, ask_ollama, build_messages, format_citation, main


class FakeVector:
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

    def search(self, _query_vector, k):
        self.searched_k = k
        return [[0.91, 0.82]], [[1, 0]]


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    @staticmethod
    def read():
        return b'{"message": {"content": "Respuesta basada en la fuente [1]."}}'


class ChatTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {
                "source": "data/Reglamento-Europeo-IA.md",
                "legal_reference": "Reglamento (UE) 2024/1689",
                "article": "9",
                "section": "Artículo 9",
                "source_version": "10/11/2026",
                "text": "El sistema de gestión de riesgos.",
            },
            {
                "source": "data/LOPDGDD.md",
                "legal_reference": "BOE-A-2018-16673",
                "article": None,
                "section": "Preámbulo",
                "text": "Texto de referencia.",
            },
        ]

    def test_search_encodes_query_and_caps_k_to_index_size(self):
        index = FakeIndex()
        embedder = FakeEmbedder()
        store = RAGStore(index, self.chunks, embedder, FakeNumpy())

        results = store.search("gestión de riesgos", k=5)

        self.assertEqual(embedder.encoded_texts, ["gestión de riesgos"])
        self.assertEqual(index.searched_k, 2)
        self.assertEqual([chunk for _, chunk in results], [self.chunks[1], self.chunks[0]])

    def test_search_rejects_empty_question_and_invalid_k(self):
        store = RAGStore(FakeIndex(), self.chunks, FakeEmbedder(), FakeNumpy())

        with self.assertRaises(ValueError):
            store.search("  ")
        with self.assertRaises(ValueError):
            store.search("consulta", k=0)

    def test_prompt_includes_question_sources_and_grounding_instructions(self):
        messages = build_messages("¿Qué exige el artículo 9?", [(0.9, self.chunks[0])])

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("únicamente", messages[0]["content"])
        self.assertIn("¿Qué exige el artículo 9?", messages[1]["content"])
        self.assertIn("[1] Reglamento (UE) 2024/1689 — artículo 9", messages[1]["content"])
        self.assertIn("10/11/2026", messages[1]["content"])
        self.assertIn("El sistema de gestión de riesgos.", messages[1]["content"])

    def test_citation_uses_section_when_chunk_has_no_article(self):
        citation = format_citation(self.chunks[1])

        self.assertIn("BOE-A-2018-16673", citation)
        self.assertIn("Preámbulo", citation)
        self.assertIn("data/LOPDGDD.md", citation)

    def test_ollama_request_uses_non_streaming_chat_endpoint(self):
        messages = [{"role": "user", "content": "Pregunta"}]
        with patch("chat.urlopen", return_value=FakeResponse()) as urlopen:
            answer = ask_ollama(messages, "test-model", "http://localhost:11434/")

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(answer, "Respuesta basada en la fuente [1].")
        self.assertEqual(request.full_url, "http://localhost:11434/api/chat")
        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(payload["messages"], messages)
        self.assertFalse(payload["stream"])

    def test_terminal_loop_accepts_another_question_before_exit(self):
        args = SimpleNamespace(model="test-model", ollama_url="http://localhost:11434", top_k=5)
        store = RAGStore(FakeIndex(), self.chunks, FakeEmbedder(), FakeNumpy())
        with (
            patch("chat.parse_args", return_value=args),
            patch("chat.load_rag_store", return_value=store),
            patch("chat.input", side_effect=["primera pregunta", "segunda pregunta", "salir"]),
            patch("chat.ask_ollama", side_effect=["respuesta 1", "respuesta 2"]) as ask,
            redirect_stdout(StringIO()) as output,
        ):
            result = main([])

        self.assertEqual(result, 0)
        self.assertEqual(ask.call_count, 2)
        self.assertIn("respuesta 1", output.getvalue())
        self.assertIn("respuesta 2", output.getvalue())


if __name__ == "__main__":
    unittest.main()
