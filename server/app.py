"""
FastAPI application for the SmartGrid-Optima Energy Management Environment.

Endpoints:
    - POST /reset: Reset the environment (accepts task_id, seed)
    - POST /step: Execute an action (0-3)
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

try:
    from models import EnergyAction, EnergyObservation
    from .smartgrid_environment import SmartGridEnvironment
except ModuleNotFoundError:
    from models import EnergyAction, EnergyObservation
    from server.smartgrid_environment import SmartGridEnvironment


app = create_app(
    SmartGridEnvironment,
    EnergyAction,
    EnergyObservation,
    env_name="smartgrid_optima",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for running the server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
