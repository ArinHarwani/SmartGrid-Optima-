"""FastAPI server for the SmartGrid-Optima Energy Management Environment."""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required. Install with: uv sync") from e

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
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
