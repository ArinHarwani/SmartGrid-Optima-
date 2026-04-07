"""
SmartGrid-Optima Environment Implementation.

A 24-hour energy management simulation for Bangalore, India.
The AI agent manages solar panels, battery storage, and grid power
to minimize electricity costs.

Episode: 24 steps (1 step = 1 hour, from hour 0 to hour 23)
Actions: 0=Idle, 1=Charge, 2=Discharge, 3=Sell
"""

import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import EnergyAction, EnergyObservation, EnergyState
    from ..data import (
        BATTERY, TASKS, REWARD,
        get_grid_buy_price, get_grid_sell_price,
        get_solar_actual_kw, get_cloud_cover,
        get_home_load, normalize_reward,
    )
except ImportError:
    from models import EnergyAction, EnergyObservation, EnergyState
    from data import (
        BATTERY, TASKS, REWARD,
        get_grid_buy_price, get_grid_sell_price,
        get_solar_actual_kw, get_cloud_cover,
        get_home_load, normalize_reward,
    )


class SmartGridEnvironment(Environment):
    """
    Smart Energy Management Environment.

    Simulates a 24-hour period where an AI agent manages:
    - Solar panels (5kW peak, Bangalore bell curve)
    - Battery storage (10kWh, 92% round-trip efficiency)
    - Grid connection (BESCOM pricing, possible outages)
    - Home load (residential or commercial profile)

    The agent takes 24 actions (one per hour) to minimize total electricity cost.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    MAX_STEPS = 24

    def __init__(self):
        """Initialize the SmartGrid environment."""
        self._state = EnergyState(episode_id=str(uuid4()), step_count=0)
        self._current_hour = 0
        self._battery_kwh = 5.0
        self._task_config = None
        self._seed = None
        self._cost_with_ai = 0.0
        self._cost_no_ai = 0.0
        self._total_solar = 0.0
        self._total_grid_bought = 0.0
        self._total_grid_sold = 0.0
        self._total_battery_cycles = 0.0
        self._blackout_occurred = False
        self._actions_taken = []
        self._rewards_history = []
        self._last_action_error = None

    def reset(self, task_id: str = None, seed: int = None, **kwargs) -> EnergyObservation:
        """
        Reset the environment for a new episode.

        Args:
            task_id: One of 'residential_summer_basic', 'commercial_tod_optimization',
                     'commercial_monsoon_resilience'. Defaults to random selection.
            seed: Random seed for reproducibility.

        Returns:
            Initial EnergyObservation showing the state at hour 0.
        """
        # Select task
        if task_id is None or task_id not in TASKS:
            task_id = random.choice(list(TASKS.keys()))

        self._task_config = TASKS[task_id]
        self._seed = seed if seed is not None else random.randint(0, 999999)

        rng = random.Random(self._seed)

        # Initialize battery SoC
        soc_min = self._task_config["soc_min"]
        soc_max = self._task_config["soc_max"]
        initial_soc = rng.uniform(soc_min, soc_max)
        self._battery_kwh = initial_soc * BATTERY["capacity_kwh"]

        # Reset tracking
        self._current_hour = 0
        self._cost_with_ai = 0.0
        self._cost_no_ai = 0.0
        self._total_solar = 0.0
        self._total_grid_bought = 0.0
        self._total_grid_sold = 0.0
        self._total_battery_cycles = 0.0
        self._blackout_occurred = False
        self._actions_taken = []
        self._rewards_history = []
        self._last_action_error = None

        # Create state
        self._state = EnergyState(
            episode_id=str(uuid4()),
            step_count=0,
            task_id=task_id,
            persona=self._task_config["persona"],
        )

        # Build initial observation
        persona = self._task_config["persona"]
        cloud = get_cloud_cover(
            0, self._task_config["cloud_min"],
            self._task_config["cloud_max"], self._seed
        )
        solar = get_solar_actual_kw(0, cloud)
        grid_buy = get_grid_buy_price(persona, 0)
        grid_sell = get_grid_sell_price(persona)
        load = get_home_load(persona, 0, seed=self._seed)

        # Check grid availability for hard task
        grid_available = True
        if self._task_config["grid_outage_chance"] > 0:
            grid_available = rng.random() >= self._task_config["grid_outage_chance"]

        soc = self._battery_kwh / BATTERY["capacity_kwh"]

        return EnergyObservation(
            done=False,
            reward=None,
            hour=0,
            step_number=0,
            solar_output_kw=solar,
            cloud_cover_pct=cloud,
            battery_soc=round(soc, 4),
            battery_kwh=round(self._battery_kwh, 3),
            grid_price_buy=grid_buy,
            grid_price_sell=grid_sell,
            grid_available=grid_available,
            home_load_kw=load,
            cost_this_step=0.0,
            cost_cumulative=0.0,
            cost_no_ai_cumulative=0.0,
            task_id=self._state.task_id,
            persona=persona,
            message=f"Episode started: {self._task_config['description']} "
                    f"Battery at {soc*100:.0f}%.",
            last_action_error=None,
        )

    def step(self, action: EnergyAction) -> EnergyObservation:
        """
        Execute one hour of energy management.

        Args:
            action: EnergyAction with action field (0-3)

        Returns:
            EnergyObservation with updated state and reward
        """
        if self._task_config is None:
            return EnergyObservation(
                done=True, reward=0.0,
                message="Error: Environment not reset. Call reset() first.",
                last_action_error="Environment not reset",
            )

        hour = self._current_hour
        persona = self._task_config["persona"]
        act = action.action

        self._actions_taken.append(act)
        self._state.step_count += 1
        self._last_action_error = None

        # --- 1. Get current conditions ---
        cloud = get_cloud_cover(
            hour, self._task_config["cloud_min"],
            self._task_config["cloud_max"], self._seed
        )
        solar_kw = get_solar_actual_kw(hour, cloud)
        grid_buy = get_grid_buy_price(persona, hour)
        grid_sell = get_grid_sell_price(persona)
        load_kw = get_home_load(persona, hour, seed=self._seed)

        # Grid availability
        rng = random.Random(self._seed + hour + 5000)
        grid_available = True
        if self._task_config["grid_outage_chance"] > 0:
            grid_available = rng.random() >= self._task_config["grid_outage_chance"]

        self._total_solar += solar_kw

        # --- 2. Execute action and compute cost ---
        cost_ai, msg = self._execute_action(
            act, solar_kw, load_kw, grid_buy, grid_sell, grid_available
        )

        # --- 3. Compute baseline (no AI) cost ---
        cost_no_ai = self._compute_no_ai_cost(
            solar_kw, load_kw, grid_buy, grid_available
        )

        # --- 4. Accumulate costs ---
        self._cost_with_ai += cost_ai
        self._cost_no_ai += cost_no_ai

        # --- 5. Compute reward ---
        raw_savings = cost_no_ai - cost_ai

        # Blackout penalty check
        soc = self._battery_kwh / BATTERY["capacity_kwh"]
        blackout_penalty = 0.0
        if not grid_available and soc < BATTERY["min_soc"] and load_kw > solar_kw:
            blackout_penalty = REWARD["blackout_penalty"]
            self._blackout_occurred = True
            msg += " ⚡ BLACKOUT! Load not met during grid outage."

        # Cycling penalty (penalize if agent charges and discharges every other step)
        cycling_penalty = 0.0
        if len(self._actions_taken) >= 3:
            last3 = self._actions_taken[-3:]
            if (last3[0] in [1, 2] and last3[1] in [1, 2] and last3[2] in [1, 2]
                    and last3[0] != last3[1] and last3[1] != last3[2]):
                cycling_penalty = REWARD["cycling_penalty"]
                self._total_battery_cycles += 0.5

        raw_reward = raw_savings + blackout_penalty + cycling_penalty
        normalized_reward = normalize_reward(raw_reward)

        self._rewards_history.append(normalized_reward)

        # --- 6. Advance hour ---
        self._current_hour += 1
        done = self._current_hour >= self.MAX_STEPS

        # --- 7. Get next hour conditions for observation ---
        next_hour = self._current_hour if not done else 23
        next_cloud = get_cloud_cover(
            next_hour, self._task_config["cloud_min"],
            self._task_config["cloud_max"], self._seed
        )
        next_solar = get_solar_actual_kw(next_hour, next_cloud) if not done else 0.0
        next_grid_buy = get_grid_buy_price(persona, next_hour)
        next_load = get_home_load(persona, next_hour, seed=self._seed) if not done else 0.0

        next_grid_available = True
        if self._task_config["grid_outage_chance"] > 0 and not done:
            next_rng = random.Random(self._seed + next_hour + 5000)
            next_grid_available = next_rng.random() >= self._task_config["grid_outage_chance"]

        soc = self._battery_kwh / BATTERY["capacity_kwh"]

        # Update state
        self._state.total_cost_with_ai = round(self._cost_with_ai, 2)
        self._state.total_cost_no_ai = round(self._cost_no_ai, 2)
        self._state.total_solar_generated = round(self._total_solar, 3)
        self._state.total_grid_bought = round(self._total_grid_bought, 3)
        self._state.total_grid_sold = round(self._total_grid_sold, 3)
        self._state.total_battery_cycles = round(self._total_battery_cycles, 2)
        self._state.blackout_occurred = self._blackout_occurred
        self._state.actions_taken = list(self._actions_taken)
        self._state.rewards_history = list(self._rewards_history)

        if done:
            total_savings = self._cost_no_ai - self._cost_with_ai
            msg += (f" | Episode complete! AI cost: ₹{self._cost_with_ai:.2f}, "
                    f"Baseline: ₹{self._cost_no_ai:.2f}, "
                    f"Savings: ₹{total_savings:.2f}")

        return EnergyObservation(
            done=done,
            reward=normalized_reward,
            hour=next_hour,
            step_number=self._current_hour,
            solar_output_kw=next_solar,
            cloud_cover_pct=next_cloud,
            battery_soc=round(soc, 4),
            battery_kwh=round(self._battery_kwh, 3),
            grid_price_buy=next_grid_buy,
            grid_price_sell=grid_sell,
            grid_available=next_grid_available,
            home_load_kw=next_load,
            cost_this_step=round(cost_ai, 2),
            cost_cumulative=round(self._cost_with_ai, 2),
            cost_no_ai_cumulative=round(self._cost_no_ai, 2),
            task_id=self._state.task_id,
            persona=persona,
            message=msg,
            last_action_error=self._last_action_error,
        )

    def _execute_action(
        self, action: int, solar_kw: float, load_kw: float,
        grid_buy: float, grid_sell: float, grid_available: bool
    ) -> tuple:
        """
        Execute an energy management action and return (cost, message).
        """
        cap = BATTERY["capacity_kwh"]
        max_charge = BATTERY["max_charge_rate_kw"]
        max_discharge = BATTERY["max_discharge_rate_kw"]
        charge_eff = BATTERY["charge_efficiency"]
        discharge_eff = BATTERY["discharge_efficiency"]
        min_soc_kwh = BATTERY["min_soc"] * cap

        cost = 0.0
        msg = ""

        # Net load after solar
        net_load = load_kw - solar_kw  # Positive = need more, Negative = excess solar

        if action == 0:  # IDLE
            if net_load > 0:
                # Need energy from grid
                if grid_available:
                    cost = net_load * grid_buy
                    self._total_grid_bought += net_load
                    msg = f"Hour {self._current_hour}: Idle. Solar covers {solar_kw:.1f}kW, "
                    msg += f"buying {net_load:.1f}kW from grid at ₹{grid_buy}"
                else:
                    # Grid down, try battery
                    available = max(0, self._battery_kwh - min_soc_kwh)
                    discharge = min(net_load / discharge_eff, available, max_discharge)
                    delivered = discharge * discharge_eff
                    self._battery_kwh -= discharge
                    shortfall = net_load - delivered
                    if shortfall > 0:
                        self._last_action_error = f"Grid down, battery low. Shortfall: {shortfall:.1f}kW"
                        msg = f"Hour {self._current_hour}: Grid DOWN! Battery supplied {delivered:.1f}kW, shortfall {shortfall:.1f}kW"
                    else:
                        msg = f"Hour {self._current_hour}: Grid DOWN. Battery supplied {delivered:.1f}kW"
            else:
                # Excess solar, just waste it
                msg = f"Hour {self._current_hour}: Idle. Solar excess of {-net_load:.1f}kW wasted."

        elif action == 1:  # CHARGE BATTERY
            if net_load <= 0:
                # Excess solar available for charging
                excess = -net_load
                charge_amount = min(excess, max_charge)
                stored = charge_amount * charge_eff
                space = cap - self._battery_kwh
                actual_stored = min(stored, space)
                actual_charge = actual_stored / charge_eff if charge_eff > 0 else 0
                self._battery_kwh += actual_stored
                self._total_battery_cycles += actual_stored / cap
                msg = f"Hour {self._current_hour}: Charging battery with {actual_stored:.1f}kWh from solar (free)"
                cost = 0.0
            else:
                # Not enough solar; charge from grid if available
                if grid_available:
                    # Use solar for load first, then charge from grid
                    cost = net_load * grid_buy  # Pay for load deficit
                    self._total_grid_bought += net_load

                    charge_amount = min(max_charge, cap - self._battery_kwh)
                    if charge_amount > 0:
                        stored = charge_amount * charge_eff
                        space = cap - self._battery_kwh
                        actual_stored = min(stored, space)
                        actual_charge = actual_stored / charge_eff if charge_eff > 0 else 0
                        cost += actual_charge * grid_buy
                        self._total_grid_bought += actual_charge
                        self._battery_kwh += actual_stored
                        self._total_battery_cycles += actual_stored / cap
                        msg = f"Hour {self._current_hour}: Charging {actual_stored:.1f}kWh from grid at ₹{grid_buy}/kWh"
                    else:
                        msg = f"Hour {self._current_hour}: Battery full, bought grid for load"
                else:
                    self._last_action_error = "Cannot charge from grid: grid is down"
                    # Just try to handle load from battery
                    available = max(0, self._battery_kwh - min_soc_kwh)
                    discharge = min(net_load / discharge_eff, available, max_discharge)
                    delivered = discharge * discharge_eff
                    self._battery_kwh -= discharge
                    msg = f"Hour {self._current_hour}: Grid DOWN, can't charge. Battery supplied {delivered:.1f}kW"

        elif action == 2:  # DISCHARGE BATTERY
            available = max(0, self._battery_kwh - min_soc_kwh)
            if available <= 0:
                self._last_action_error = "Battery at minimum SoC, cannot discharge"
                # Fall back to grid/solar
                if net_load > 0:
                    if grid_available:
                        cost = net_load * grid_buy
                        self._total_grid_bought += net_load
                        msg = f"Hour {self._current_hour}: Battery empty! Buying {net_load:.1f}kW from grid"
                    else:
                        msg = f"Hour {self._current_hour}: Battery empty, grid DOWN! Blackout risk!"
                        self._last_action_error = "No power source available"
                else:
                    msg = f"Hour {self._current_hour}: Battery at min. Solar covers load."
            else:
                # Discharge to cover load
                needed = max(0, net_load)
                discharge = min(needed / discharge_eff if needed > 0 else 0, available, max_discharge)
                delivered = discharge * discharge_eff
                self._battery_kwh -= discharge
                self._total_battery_cycles += discharge / cap

                remaining_load = max(0, needed - delivered)
                if remaining_load > 0 and grid_available:
                    cost = remaining_load * grid_buy
                    self._total_grid_bought += remaining_load
                    msg = f"Hour {self._current_hour}: Discharged {delivered:.1f}kW, "
                    msg += f"grid supplied {remaining_load:.1f}kW at ₹{grid_buy}"
                elif remaining_load > 0:
                    msg = f"Hour {self._current_hour}: Discharged {delivered:.1f}kW, "
                    msg += f"grid DOWN, shortfall {remaining_load:.1f}kW"
                    self._last_action_error = f"Shortfall: {remaining_load:.1f}kW"
                else:
                    savings = needed * grid_buy if needed > 0 else 0
                    msg = f"Hour {self._current_hour}: Discharged {delivered:.1f}kW. "
                    msg += f"Saved ₹{savings:.1f} vs grid"

        elif action == 3:  # SELL TO GRID
            if not grid_available:
                self._last_action_error = "Cannot sell: grid is down"
                # Handle load normally
                if net_load > 0:
                    available = max(0, self._battery_kwh - min_soc_kwh)
                    discharge = min(net_load / discharge_eff, available, max_discharge)
                    delivered = discharge * discharge_eff
                    self._battery_kwh -= discharge
                    msg = f"Hour {self._current_hour}: Grid DOWN, can't sell. Battery supplied {delivered:.1f}kW"
                else:
                    msg = f"Hour {self._current_hour}: Grid DOWN. Solar covers load, can't sell excess"
            elif net_load < 0:
                # Sell excess solar
                excess = -net_load
                sell_amount = excess
                revenue = sell_amount * grid_sell
                cost = -revenue  # Negative cost = income
                self._total_grid_sold += sell_amount
                msg = f"Hour {self._current_hour}: Sold {sell_amount:.1f}kW solar at ₹{grid_sell}/kWh. "
                msg += f"Earned ₹{revenue:.1f}"
            else:
                # No excess solar, nothing to sell meaningfully
                # Handle load from grid
                cost = net_load * grid_buy
                self._total_grid_bought += net_load
                self._last_action_error = "No excess energy to sell"
                msg = f"Hour {self._current_hour}: No excess to sell. Bought {net_load:.1f}kW from grid"

        return cost, msg

    def _compute_no_ai_cost(
        self, solar_kw: float, load_kw: float,
        grid_buy: float, grid_available: bool
    ) -> float:
        """
        Compute baseline cost without AI (naive strategy: use solar, buy rest from grid).
        """
        net_load = load_kw - solar_kw
        if net_load > 0:
            if grid_available:
                return net_load * grid_buy
            else:
                return net_load * grid_buy * 1.5  # Penalty for unmet load
        return 0.0

    @property
    def state(self) -> EnergyState:
        """Get the current environment state."""
        return self._state


__all__ = ["SmartGridEnvironment"]
