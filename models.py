"""
Data models for the SmartGrid-Optima Energy Management Environment.

Action Space: 4 discrete actions for energy management
Observation Space: Full state of solar, battery, grid, and home load
State: Episode tracking with cost comparison metrics
"""

from typing import List, Optional

from pydantic import Field

from openenv.core.env_server.types import Action, Observation, State


class EnergyAction(Action):
    """
    Action for the SmartGrid-Optima environment.

    Action Space:
        0 = Idle       — Do nothing, let solar handle load, buy from grid if needed
        1 = Charge     — Charge battery from solar (priority) or grid
        2 = Discharge  — Discharge battery to power home load
        3 = Sell       — Sell excess solar/battery power to grid
    """

    action: int = Field(
        ...,
        ge=0,
        le=3,
        description="Energy management action: 0=Idle, 1=Charge, 2=Discharge, 3=Sell",
    )


class EnergyObservation(Observation):
    """
    Observation from the SmartGrid-Optima environment.

    Provides complete visibility into the current energy state.
    """

    # Time
    hour: int = Field(default=0, description="Current hour of day (0-23)")
    step_number: int = Field(default=0, description="Current step in episode (0-23)")

    # Solar
    solar_output_kw: float = Field(
        default=0.0, description="Current solar panel output in kW"
    )
    cloud_cover_pct: float = Field(
        default=0.0, description="Cloud coverage percentage (0-100)"
    )

    # Battery
    battery_soc: float = Field(
        default=0.5,
        description="Battery state of charge (0.0-1.0, where 1.0 = full 10kWh)",
    )
    battery_kwh: float = Field(
        default=5.0, description="Battery energy stored in kWh (0-10)"
    )

    # Grid
    grid_price_buy: float = Field(
        default=5.90, description="Current grid buy price in ₹/kWh"
    )
    grid_price_sell: float = Field(
        default=3.86, description="Current grid sell price in ₹/kWh"
    )
    grid_available: bool = Field(default=True, description="Is grid power available")

    # Home
    home_load_kw: float = Field(
        default=1.0, description="Current home power consumption in kW"
    )

    # Cost tracking
    cost_this_step: float = Field(
        default=0.0, description="Cost incurred this step in ₹"
    )
    cost_cumulative: float = Field(
        default=0.0, description="Total cost so far in ₹"
    )
    cost_no_ai_cumulative: float = Field(
        default=0.0, description="Baseline cost without AI in ₹"
    )

    # Task info
    task_id: str = Field(default="", description="Current task identifier")
    persona: str = Field(
        default="residential", description="Pricing persona: residential or commercial"
    )

    # Status
    message: str = Field(default="", description="Human-readable status message")
    last_action_error: Optional[str] = Field(
        default=None, description="Error message from last action, or null"
    )


class EnergyState(State):
    """
    Extended state for SmartGrid-Optima environment.

    Tracks episode-level metrics beyond what Observation provides.
    """

    task_id: str = Field(default="", description="Active task identifier")
    persona: str = Field(default="residential", description="Pricing persona")
    total_cost_no_ai: float = Field(
        default=0.0, description="Baseline cost without AI management"
    )
    total_cost_with_ai: float = Field(
        default=0.0, description="AI-managed cost"
    )
    total_solar_generated: float = Field(
        default=0.0, description="Total solar energy generated in kWh"
    )
    total_grid_bought: float = Field(
        default=0.0, description="Total grid energy purchased in kWh"
    )
    total_grid_sold: float = Field(
        default=0.0, description="Total energy sold to grid in kWh"
    )
    total_battery_cycles: float = Field(
        default=0.0, description="Total battery charge/discharge cycles"
    )
    blackout_occurred: bool = Field(
        default=False, description="Whether a blackout happened"
    )
    actions_taken: List[int] = Field(
        default_factory=list, description="History of actions taken"
    )
    rewards_history: List[float] = Field(
        default_factory=list, description="History of rewards received"
    )


__all__ = [
    "EnergyAction",
    "EnergyObservation",
    "EnergyState",
]
