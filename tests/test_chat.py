import os
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

from chat import main, parse_args


class ChatTests(unittest.TestCase):
    def test_terminal_uses_shared_connection_for_successive_questions(self):
        args = SimpleNamespace(model="test-model", ollama_url="http://localhost:11434", top_k=5)
        store = object()
        chunk = {"source": "norma.md", "text": "Texto de referencia."}
        with (
            patch("chat.parse_args", return_value=args),
            patch("rag.load_rag_store", return_value=store),
            patch("chat.input", side_effect=["primera pregunta", "segunda pregunta", "salir"]),
            patch(
                "rag.answer_question",
                side_effect=[
                    ("respuesta 1", [(0.9, chunk)]),
                    ("respuesta 2", [(0.8, chunk)]),
                ],
            ) as answer,
            redirect_stdout(StringIO()) as output,
        ):
            result = main([])

        self.assertEqual(result, 0)
        self.assertEqual(answer.call_count, 2)
        self.assertEqual(
            answer.call_args_list[0].args,
            (store, "primera pregunta", 5, "test-model", "http://localhost:11434"),
        )
        self.assertIn("respuesta 1", output.getvalue())
        self.assertIn("respuesta 2", output.getvalue())
        self.assertIn("Fuentes recuperadas:", output.getvalue())

    def test_missing_store_reports_error_and_does_not_enter_chat(self):
        with (
            patch("rag.load_rag_store", side_effect=RuntimeError("Ejecuta crear_rag.ipynb")),
            patch("chat.input") as user_input,
            redirect_stderr(StringIO()) as output,
        ):
            result = main([])

        self.assertEqual(result, 1)
        user_input.assert_not_called()
        self.assertIn("crear_rag.ipynb", output.getvalue())

    def test_ollama_failure_is_reported_and_next_question_is_allowed(self):
        with (
            patch("rag.load_rag_store", return_value=object()),
            patch("chat.input", side_effect=["pregunta", "otra pregunta", "salir"]),
            patch(
                "rag.answer_question",
                side_effect=[RuntimeError("No se pudo conectar con Ollama"), ("respuesta", [])],
            ) as answer,
            redirect_stdout(StringIO()) as output,
        ):
            self.assertEqual(main([]), 0)

        self.assertEqual(answer.call_count, 2)
        self.assertIn("Error: No se pudo conectar con Ollama", output.getvalue())
        self.assertIn("respuesta", output.getvalue())

    def test_environment_configuration_and_cli_overrides(self):
        with patch.dict(
            os.environ, {"OLLAMA_MODEL": "local-model", "OLLAMA_URL": "http://localhost:1234"}
        ):
            args = parse_args([])
            override = parse_args(["--model", "other-model", "--top-k", "2"])

        self.assertEqual(args.model, "local-model")
        self.assertEqual(args.ollama_url, "http://localhost:1234")
        self.assertEqual(override.model, "other-model")
        self.assertEqual(override.top_k, 2)


if __name__ == "__main__":
    unittest.main()
