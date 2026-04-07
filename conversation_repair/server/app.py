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
    # Normal case (when imported as module)
    from conversation_repair.models import ConversationRepairAction, ConversationRepairObservation
    from conversation_repair.server.conversation_repair_environment import ConversationRepairEnvironment
except Exception:
    # Fallback (when run as script: python server/app.py)
    from models import ConversationRepairAction, ConversationRepairObservation
    from conversation_repair_environment import ConversationRepairEnvironment


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


# ✅ Main function must be callable
def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for direct execution via `python server/app.py`."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


# ✅ REQUIRED for evaluator
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    main(port=args.port)