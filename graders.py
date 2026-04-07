"""
Graders for SmartGrid-Optima tasks.

Three grader functions that evaluate agent performance on a 0.0-1.0 scale.
Each grader runs a complete 24-step episode and compares AI cost vs baseline.
"""

from typing import Dict, Tuple

try:
    from .data import TASKS, BATTERY, normalize_reward
    from .server.smartgrid_environment import SmartGridEnvironment
    from .models import EnergyAction
except ImportError:
    from data import TASKS, BATTERY, normalize_reward
    from server.smartgrid_environment import SmartGridEnvironment
    from models import EnergyAction


def run_episode_with_actions(task_id: str, actions: list, seed: int = 42) -> Dict:
    """
    Run a complete episode with given actions and return results.

    Args:
        task_id: Task identifier
        actions: List of 24 integer actions (0-3)
        seed: Random seed

    Returns:
        Dictionary with episode results
    """
    env = SmartGridEnvironment()
    obs = env.reset(task_id=task_id, seed=seed)

    rewards = []
    total_steps = 0

    for i, act_val in enumerate(actions):
        if i >= 24:
            break
        act = EnergyAction(action=act_val)
        obs = env.step(act)
        rewards.append(obs.reward if obs.reward is not None else 0.0)
        total_steps += 1
        if obs.done:
            break

    state = env.state
    return {
        "task_id": task_id,
        "total_steps": total_steps,
        "rewards": rewards,
        "final_score": sum(rewards) / len(rewards) if rewards else 0.0,
        "cost_with_ai": state.total_cost_with_ai,
        "cost_no_ai": state.total_cost_no_ai,
        "savings": state.total_cost_no_ai - state.total_cost_with_ai,
        "blackout": state.blackout_occurred,
        "solar_generated": state.total_solar_generated,
        "grid_bought": state.total_grid_bought,
        "grid_sold": state.total_grid_sold,
    }


def grade_residential_summer_basic(actions: list, seed: int = 42) -> float:
    """
    Grade agent performance on the EASY task.

    Evaluates cost savings on a sunny residential day.
    Score: 0.0 (worse than baseline) to 1.0 (perfect management).
    """
    results = run_episode_with_actions("residential_summer_basic", actions, seed)
    return round(results["final_score"], 4)


def grade_commercial_tod_optimization(actions: list, seed: int = 42) -> float:
    """
    Grade agent performance on the MEDIUM task.

    Evaluates ToD price arbitrage on a partly cloudy commercial day.
    Score: 0.0 to 1.0.
    """
    results = run_episode_with_actions("commercial_tod_optimization", actions, seed)
    return round(results["final_score"], 4)


def grade_commercial_monsoon_resilience(actions: list, seed: int = 42) -> float:
    """
    Grade agent performance on the HARD task.

    Evaluates monsoon resilience with grid outages on commercial ToD.
    Score: 0.0 to 1.0. Blackouts heavily penalized.
    """
    results = run_episode_with_actions("commercial_monsoon_resilience", actions, seed)

    # Additional penalty for blackout
    if results["blackout"]:
        return round(max(0.0, results["final_score"] * 0.3), 4)

    return round(results["final_score"], 4)


def grade_all(actions_per_task: Dict[str, list], seed: int = 42) -> Dict[str, float]:
    """
    Grade all three tasks and return scores.

    Args:
        actions_per_task: Dict mapping task_id to list of 24 actions
        seed: Random seed

    Returns:
        Dict mapping task_id to score (0.0-1.0)
    """
    return {
        "residential_summer_basic": grade_residential_summer_basic(
            actions_per_task.get("residential_summer_basic", [0] * 24), seed
        ),
        "commercial_tod_optimization": grade_commercial_tod_optimization(
            actions_per_task.get("commercial_tod_optimization", [0] * 24), seed
        ),
        "commercial_monsoon_resilience": grade_commercial_monsoon_resilience(
            actions_per_task.get("commercial_monsoon_resilience", [0] * 24), seed
        ),
    }


__all__ = [
    "grade_residential_summer_basic",
    "grade_commercial_tod_optimization",
    "grade_commercial_monsoon_resilience",
    "grade_all",
    "run_episode_with_actions",
]
