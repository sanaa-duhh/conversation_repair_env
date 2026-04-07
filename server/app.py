"""
FastAPI application for the `conversation_repair` environment.

This module exposes HTTP + WebSocket endpoints compatible with OpenEnv EnvClient.
"""

import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with "
        "'uv sync' (or your equivalent) in the environment root."
    ) from e

from conversation_repair.models import ConversationRepairAction, ConversationRepairObservation
from conversation_repair.server.conversation_repair_environment import ConversationRepairEnvironment


app = create_app(
    ConversationRepairEnvironment,
    ConversationRepairAction,
    ConversationRepairObservation,
    env_name="conversation_repair",
    max_concurrent_envs=1,
)


# ✅ Health + root endpoints (required)
@app.get("/")
def root():
    return {"status": "ok", "env": "conversation_repair"}


@app.get("/health")
def health():
    return {"status": "healthy"}


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)