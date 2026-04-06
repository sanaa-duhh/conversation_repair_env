"""
FastAPI application for the `conversation_repair` environment.

This module exposes HTTP + WebSocket endpoints compatible with OpenEnv EnvClient.
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with "
        "'uv sync' (or your equivalent) in the environment root."
    ) from e

try:
    from ..models import ConversationRepairAction, ConversationRepairObservation
    from .conversation_repair_environment import ConversationRepairEnvironment
except ModuleNotFoundError:
    from models import ConversationRepairAction, ConversationRepairObservation
    from server.conversation_repair_environment import ConversationRepairEnvironment


app = create_app(
    ConversationRepairEnvironment,
    ConversationRepairAction,
    ConversationRepairObservation,
    env_name="conversation_repair",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for direct execution via `uv run` or `python -m`."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)

