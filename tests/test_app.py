import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import streamlit as st
from streamlit.testing.v1 import AppTest

import rag

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class AppTests(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop("OLLAMA_MODEL", None)
        os.environ.pop("OLLAMA_URL", None)
        self.chunk = {
            "source": "data/norma.md",
            "legal_reference": "Norma de prueba",
            "article": "9",
            "chunk_id": 0,
            "text": "El sistema de gestión de riesgos.",
        }
        self.store = SimpleNamespace(
            metadata={"embedding_model": "fake-embedder", "num_documents": 1},
            chunks=[self.chunk],
            search=Mock(return_value=[(0.9, self.chunk)]),
        )

    def test_missing_store_shows_notebook_instructions_without_chat(self):
        with patch(
            "rag.load_rag_store", side_effect=RuntimeError("Ejecuta crear_rag.ipynb")
        ):
            app = AppTest.from_file(str(APP_PATH)).run()

        self.assertEqual(len(app.exception), 0)
        self.assertIn("crear_rag.ipynb", app.error[0].value)
        self.assertEqual(len(app.chat_input), 0)

    def test_frontend_uses_shared_pipeline_and_renders_citations(self):
        with (
            patch("rag.load_rag_store", return_value=self.store),
            patch("rag.ask_ollama", return_value="Respuesta generada [1].") as ask,
        ):
            app = AppTest.from_file(str(APP_PATH)).run()
            app.chat_input[0].set_value("¿Qué exige la norma?").run()

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(app.toggle[0].value)
        self.assertEqual(len(app.button), 0)
        self.store.search.assert_called_once_with("¿Qué exige la norma?", 4)
        messages, model, url = ask.call_args.args
        self.assertEqual(model, "qwen2.5:3b")
        self.assertEqual(url, "http://localhost:11434")
        self.assertIn("[1] Norma de prueba", messages[1]["content"])
        rendered = "\n".join(element.value for element in app.markdown)
        self.assertIn("Respuesta generada [1].", rendered)
        self.assertIn(rag.format_citation(self.chunk), rendered)
        self.assertEqual(len(app.session_state["messages"]), 2)

    def test_retrieval_only_is_explicit_and_never_calls_ollama(self):
        with (
            patch("rag.load_rag_store", return_value=self.store),
            patch("rag.ask_ollama") as ask,
        ):
            app = AppTest.from_file(str(APP_PATH)).run()
            app.toggle[0].set_value(False).run()
            app.chat_input[0].set_value("pregunta").run()

        self.assertEqual(len(app.exception), 0)
        ask.assert_not_called()
        self.assertIn(
            "fragmentos más relevantes", app.session_state["messages"][-1]["content"]
        )

    def test_ollama_failure_is_visible_and_is_not_saved_as_success(self):
        with (
            patch("rag.load_rag_store", return_value=self.store),
            patch("rag.ask_ollama", side_effect=RuntimeError("Ollama no disponible")),
        ):
            app = AppTest.from_file(str(APP_PATH)).run()
            app.chat_input[0].set_value("pregunta").run()

        self.assertEqual(len(app.exception), 0)
        self.assertIn("Ollama no disponible", app.error[0].value)
        self.assertEqual(
            [message["role"] for message in app.session_state["messages"]], ["user"]
        )


if __name__ == "__main__":
    unittest.main()
