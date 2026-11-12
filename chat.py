"""Terminal client for the shared RAG connection."""

from __future__ import annotations

import argparse
import os
import sys

import rag


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consulta por terminal la legislación indexada en rag_store/."
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OLLAMA_MODEL", rag.DEFAULT_MODEL),
        help=f"Modelo de Ollama (por defecto: {rag.DEFAULT_MODEL}; también OLLAMA_MODEL).",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.getenv("OLLAMA_URL", rag.DEFAULT_OLLAMA_URL),
        help="URL del servidor Ollama (también OLLAMA_URL).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=rag.DEFAULT_TOP_K,
        help=f"Número de fragmentos de contexto (por defecto: {rag.DEFAULT_TOP_K}).",
    )
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k debe ser al menos 1.")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        store = rag.load_rag_store()
    except (RuntimeError, ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(
        f"Chat de legislación listo. Modelo: {args.model}. "
        "Escribe 'salir' para terminar."
    )
    while True:
        try:
            question = input("\nConsulta> ").strip()
        except EOFError:
            print("\nFin de la entrada.")
            break
        except KeyboardInterrupt:
            print("\nSesión terminada.")
            break

        if question.lower() in {"salir", "exit", "quit"}:
            print("Sesión terminada.")
            break
        if not question:
            continue

        try:
            answer, results = rag.answer_question(
                store, question, args.top_k, args.model, args.ollama_url
            )
        except (RuntimeError, ValueError, OSError) as error:
            print(f"Error: {error}")
            continue

        print(f"\n{answer}")
        if results:
            print("\nFuentes recuperadas:")
            for citation_number, (score, chunk) in enumerate(results, start=1):
                print(
                    f"[{citation_number}] {rag.format_citation(chunk)} "
                    f"(similitud: {score:.3f})"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
