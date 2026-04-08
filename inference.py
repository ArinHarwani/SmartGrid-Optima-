import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from client import SmartGridEnv
from models import EnergyAction

IMAGE_NAME = os.getenv("IMAGE_NAME")  
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = HF_TOKEN or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://api.openai.com/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "gpt-4.1-mini"

TASK_NAME = os.getenv("SMARTGRID_TASK", "commercial_monsoon_resilience")
BENCHMARK = os.getenv("SMARTGRID_BENCHMARK", "smartgrid_optima")
MAX_STEPS = 24
TEMPERATURE = 0.1
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.5  

_MAX_REWARD_PER_STEP = 1.0
MAX_TOTAL_REWARD = MAX_STEPS * _MAX_REWARD_PER_STEP

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert AI energy manager for a building in Bangalore, India.
    Your goal: MINIMIZE electricity cost over 24 hours by choosing the best action each hour.
    
    ACTION SPACE:
    0: Idle (let solar handle load; buy from/sell to grid natively)
    1: Charge (pull energy into battery from solar/grid)
    2: Discharge (push energy from battery to handle load)
    3: Sell (push excess energy to grid for revenue)
    
    Reply with exactly one digit — 0, 1, 2, or 3.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_user_prompt(step: int, obs, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}/24 (Hour {obs.hour})
        Solar: {obs.solar_output_kw:.1f}kW | Load: {obs.home_load_kw:.1f}kW | Battery: {obs.battery_soc*100:.0f}%
        Grid Buy: ₹{obs.grid_price_buy:.2f} | Grid Available: {obs.grid_available}
        Cost So Far: ₹{obs.cost_cumulative:.2f}
        Previous steps:
        {history_block}
        What is your next action?
        """
    ).strip()


def get_model_message(client: OpenAI, step: int, obs, history: List[str]) -> str:
    user_prompt = build_user_prompt(step, obs, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        for char in text:
            if char in "0123":
                return char
        return "0"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "0"


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = await SmartGridEnv.from_docker_image(IMAGE_NAME)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_res = await env.reset()
        
        obs = reset_res.observation if hasattr(reset_res, 'observation') else reset_res

        action_names = {0: "Idle", 1: "Charge", 2: "Discharge", 3: "Sell"}
        result = reset_res  # Track current result for done-check

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action_str = get_model_message(client, step, obs, history)
            action_int = int(action_str)
            action_msg = action_names.get(action_int, "Idle")

            action_obj = EnergyAction(action=action_int)
            result = await env.step(action_obj)
            
            obs = result.observation if hasattr(result, 'observation') else result
            reward = result.reward if hasattr(result, 'reward') else getattr(obs, "reward", 0.0)
            done = result.done if hasattr(result, 'done') else getattr(obs, "done", False)
            error = getattr(result, "last_action_error", None) or getattr(obs, "last_action_error", None)

            reward = reward if reward is not None else 0.0

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            history.append(f"Step {step}: {action_msg!r} -> reward {reward:+.2f}")

            if done:
                break

        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0) 
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
