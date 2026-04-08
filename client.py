"""SmartGrid-Optima Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from models import EnergyAction, EnergyObservation, EnergyState


class SmartGridEnv(
    EnvClient[EnergyAction, EnergyObservation, EnergyState]
):
    """
    Client for the SmartGrid-Optima Environment.

    Maintains a persistent WebSocket connection to the environment server.

    Example:
        >>> with SmartGridEnv(base_url="http://localhost:8000").sync() as client:
        ...     result = client.reset()
        ...     print(result.observation.solar_output_kw)
        ...     result = client.step(EnergyAction(action=1))
        ...     print(result.observation.battery_soc)
    """

    def _step_payload(self, action: EnergyAction) -> Dict:
        """Convert EnergyAction to JSON payload."""
        return {"action": action.action}

    def _parse_result(self, payload: Dict) -> StepResult[EnergyObservation]:
        """Parse server response into StepResult[EnergyObservation]."""
        obs_data = payload.get("observation", {})
        observation = EnergyObservation(
            done=payload.get("done", False),
            reward=payload.get("reward"),
            hour=obs_data.get("hour", 0),
            step_number=obs_data.get("step_number", 0),
            solar_output_kw=obs_data.get("solar_output_kw", 0.0),
            cloud_cover_pct=obs_data.get("cloud_cover_pct", 0.0),
            battery_soc=obs_data.get("battery_soc", 0.5),
            battery_kwh=obs_data.get("battery_kwh", 5.0),
            grid_price_buy=obs_data.get("grid_price_buy", 5.90),
            grid_price_sell=obs_data.get("grid_price_sell", 3.86),
            grid_available=obs_data.get("grid_available", True),
            home_load_kw=obs_data.get("home_load_kw", 1.0),
            cost_this_step=obs_data.get("cost_this_step", 0.0),
            cost_cumulative=obs_data.get("cost_cumulative", 0.0),
            cost_no_ai_cumulative=obs_data.get("cost_no_ai_cumulative", 0.0),
            task_id=obs_data.get("task_id", ""),
            persona=obs_data.get("persona", "residential"),
            message=obs_data.get("message", ""),
            last_action_error=obs_data.get("last_action_error"),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> EnergyState:
        """Parse server response into EnergyState."""
        return EnergyState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            persona=payload.get("persona", "residential"),
            total_cost_no_ai=payload.get("total_cost_no_ai", 0.0),
            total_cost_with_ai=payload.get("total_cost_with_ai", 0.0),
            total_solar_generated=payload.get("total_solar_generated", 0.0),
            total_grid_bought=payload.get("total_grid_bought", 0.0),
            total_grid_sold=payload.get("total_grid_sold", 0.0),
            total_battery_cycles=payload.get("total_battery_cycles", 0.0),
            blackout_occurred=payload.get("blackout_occurred", False),
            actions_taken=payload.get("actions_taken", []),
            rewards_history=payload.get("rewards_history", []),
        )
