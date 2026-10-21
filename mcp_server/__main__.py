"""Serve the claim tools over stdio for any MCP client.

    python -m mcp_server [--actor NAME]

The database comes from APP_DATABASE_URL (the claimcheck_app role) and the tools read as an
adjuster named `--actor`, which is also the name their calls are audited under. The embedding
model is the local one the clauses were indexed with. An MCP client that lets a person approve
tool calls is what makes `human_confirmed` mean something here.
"""

import argparse
from collections.abc import Sequence

from core.settings import get_settings
from core.tracing import setup_tracing
from db.session import engine_for
from mcp_server.server import build_server, tool_sessions
from rag.embedder import FastEmbedEmbedder


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m mcp_server", description=__doc__)
    parser.add_argument(
        "--actor", default="mcp-client", help="the adjuster the tools act for, as audited"
    )
    args = parser.parse_args(argv)
    settings = get_settings()
    setup_tracing("claimcheck-mcp", settings.otel_endpoint)
    sessions = tool_sessions(engine_for(settings.app_database_url), args.actor)
    build_server(sessions, FastEmbedEmbedder(), args.actor).run("stdio")


if __name__ == "__main__":
    main()
