"""Command-line interface for the Local Context Engine testbed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .engine import LocalContextEngine
from .models import SourceKind
from .providers import to_deepseek_payload, to_openai_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="local-context-engine",
        description="Local Context Engine: retrieval/compression/prioritization before any LLM API.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Ingest files/history/trajectory into a local index.")
    ingest.add_argument("path", help="File, directory, .jsonl history, or .jsonl trajectory.")
    ingest.add_argument("--index", default=".lce_index.json", help="Index file path.")
    ingest.add_argument("--backend", choices=["auto", "json", "sqlite"], default="auto", help="Index backend.")
    ingest.add_argument("--kind", choices=[k.value for k in SourceKind], default=None, help="Force source kind.")
    ingest.add_argument("--as-trajectory", action="store_true", help="Treat JSONL as agent trajectory.")
    ingest.add_argument("--extensions", nargs="*", default=None, help="Extensions to ingest from a directory.")
    ingest.add_argument("--incremental", action="store_true", help="Only ingest new/changed files (directory/file).")

    build = sub.add_parser("build", help="Build a dynamic context window for a query.")
    build.add_argument("query", help="The current user/agent query.")
    build.add_argument("--index", default=".lce_index.json", help="Index file path.")
    build.add_argument("--backend", choices=["auto", "json", "sqlite"], default="auto", help="Index backend.")
    build.add_argument("--max-tokens", type=int, default=8000, help="Total context window size.")
    build.add_argument("--reserved", type=int, default=1024, help="Tokens reserved for model output.")
    build.add_argument("--format", choices=["text", "json", "deepseek", "openai"], default="text", help="Output format.")
    build.add_argument("--model", default="deepseek-chat", help="Model name for deepseek/openai payloads.")
    build.add_argument("--system", default=None, help="System prompt override for deepseek/openai payloads.")
    build.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for deepseek/openai payloads.")
    build.add_argument("--stream", action="store_true", help="Set stream=true in the API payload.")

    demo = sub.add_parser("demo", help="Run the built-in end-to-end demo.")
    demo.add_argument("--index", default=".lce_demo_index.json", help="Index file path.")
    demo.add_argument("--backend", choices=["auto", "json", "sqlite"], default="auto", help="Index backend.")

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "ingest":
        return _cmd_ingest(args)
    if args.command == "build":
        return _cmd_build(args)
    if args.command == "demo":
        return _cmd_demo(args)
    return 1


def _cmd_ingest(args) -> int:
    engine = LocalContextEngine(storage_path=args.index, backend=args.backend)
    path = Path(args.path)
    kind = SourceKind(args.kind) if args.kind else None
    count = 0
    if path.is_dir():
        if args.incremental:
            count, _changed, _skipped = engine.add_directory_incremental(
                path, kind=kind, extensions=args.extensions
            )
        else:
            count = engine.add_directory(path, kind=kind, extensions=args.extensions)
    elif path.suffix.lower() == ".jsonl" and args.as_trajectory:
        count = engine.add_trajectory_jsonl(path)
    elif path.suffix.lower() == ".jsonl":
        count = engine.add_history_jsonl(path)
    else:
        if args.incremental:
            count, _changed, _skipped = engine.add_file_incremental(path, kind=kind)
        else:
            count = engine.add_file(path, kind=kind)
    engine.save()
    print(f"Ingested {count} chunks into {args.index}")
    return 0


def _cmd_build(args) -> int:
    engine = LocalContextEngine(storage_path=args.index, backend=args.backend)
    window = engine.build_context(
        query=args.query,
        max_tokens=args.max_tokens,
        reserved_output_tokens=args.reserved,
    )
    if args.format == "json":
        print(json.dumps(window.report(), ensure_ascii=False, indent=2))
    elif args.format == "deepseek":
        payload = to_deepseek_payload(
            window,
            model=args.model,
            system_prompt=args.system,
            temperature=args.temperature,
            stream=args.stream,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif args.format == "openai":
        payload = to_openai_payload(
            window,
            model=args.model,
            system_prompt=args.system,
            temperature=args.temperature,
            stream=args.stream,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(window.text)
        print(f"\n---\nTokens: {window.total_tokens}/{window.token_budget} "
              f"(omitted {window.omitted_count} items, ~{window.omitted_tokens} tokens)")
    return 0


def _cmd_demo(args) -> int:
    from .demo import run_demo

    run_demo(index_path=args.index, backend=args.backend)
    return 0


if __name__ == "__main__":
    sys.exit(main())
