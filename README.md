---
title: SmartGrid-Optima
emoji: ⚡
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
---

<div align="center">
  <img src="https://img.shields.io/badge/Meta%20PyTorch-Hackathon-blue?style=for-the-badge&logo=pytorch" alt="Meta PyTorch Hackathon">
  <img src="https://img.shields.io/badge/Hugging%20Face-OpenEnv-yellow?style=for-the-badge&logo=huggingface" alt="Hugging Face OpenEnv">
  
  <h1>⚡ SmartGrid-Optima 🔋☀️</h1>
  <p><strong>A Next-Generation Reinforcement Learning Environment for Smart Energy Management</strong></p>
  <i>Built for the 2026 Meta PyTorch OpenEnv Hackathon</i>
</div>

---

## 🌟 Vision & Purpose

**SmartGrid-Optima** is an advanced, high-fidelity Reinforcement Learning (RL) simulation built on the Meta PyTorch **OpenEnv** framework. It is designed to train, evaluate, and benchmark AI agents acting as intelligent "Smart Energy Managers" for buildings in Bangalore, India. 

As the world rapidly transitions to renewable energy, global electricity grids face immense pressure. In rapidly growing technological hubs like Bangalore, energy demands fluctuate wildly due to localized weather patterns, rolling blackout events, and rigid Time-of-Day (ToD) pricing mandates.

**By mastering this environment, an AI agent proves it can:**
- **Slash electricity bills by 30-50%** by actively arbitraging peak and off-peak energy pricing.
- **Guarantee Grid Resilience** by proactively reserving battery storage to survive severe monsoon blackouts.
- **Accelerate Net-Zero goals** by intelligently maximizing volatile solar energy consumption.

---

## 🧠 Core Concept: Reinforcement Learning (RL)

At its absolute core, **Reinforcement Learning** is a branch of Artificial Intelligence where an agent learns to make optimal decisions by continuously interacting with an external environment. 

In the SmartGrid-Optima benchmark, Large Language Models (LLMs) act as the "brain" of the agent, and the loop operates dynamically over a 24-hour simulation:

1. 👁️ **Observation (State):** The AI observes the current physics of the house (*“It is 2:00 PM, the sun is shining at 5kW, the battery is 50% charged, and grid electricity costs ₹8.00/kWh.”*)
2. ⚡ **Action:** The AI calculates a logical strategy and executes a command (*"Charge the battery using the free solar power!"*).
3. 💰 **Reward:** The physics engine fast-forwards an hour and scores the AI based on its efficiency (*"+1.5 points for saving money, or -100 points if the house lost power."*)

---

## 🎮 The Mechanics

### 1. Action Space
Each hour, the AI must output exactly one discrete decision:
| Action | Name | Description |
|--------|------|-------------|
| `0` | **Idle** | Let the solar panels handle the load natively; buy whatever deficit remains directly from the grid. |
| `1` | **Charge** | Actively siphon free solar power (or cheap grid power) directly into the battery for later use. |
| `2` | **Discharge** | Deploy battery power into the house to intentionally avoid buying extremely expensive grid electricity. |
| `3` | **Sell** | Export excess solar or battery power back onto the government grid for revenue. |

### 2. Observation Space
The AI receives a rich, telemetry-dense payload every hour:
*   **Time & Weather:** `hour`, `solar_output_kw`, `cloud_cover_pct`
*   **Physics Status:** `battery_soc`, `battery_kwh`, `home_load_kw`
*   **Grid Economics:** `grid_price_buy`, `grid_price_sell`, `grid_available` (Outage tracker)
*   **Performance:** `cost_cumulative`

---

## 📋 Evaluation Benchmarks (Tasks)

Agents evaluating on this environment are subjected to three progressing difficulty tiers:

1. 🟢 **`residential_summer_basic` (Easy):** A perfect, sunny summer day with flat consumer pricing. The LLM simply needs to learn to harvest free solar energy.
2. 🟡 **`commercial_tod_optimization` (Medium):** A cloudy business day utilizing rigid Time-of-Day pricing tariffs. The LLM must learn to buy power at night and discharge it during peak afternoon hours.
3. 🔴 **`commercial_monsoon_resilience` (Hard):** A brutal monsoon storm. Solar output is nearly zero, and there is a 15% chance of rolling grid blackouts every hour. The LLM must carefully ration its battery; if the house runs out of power during an outage, the agent suffers a massive -100 point penalty.

---

## 🏗️ Architecture & Physics Engine

SmartGrid-Optima enforces realistic constraints verified against hardware specifications and BESCOM (Bangalore Electricity Supply Company) pricing models.

*   **Solar System:** 5kW peak, modeled directly utilizing a generic Gaussian Bangalore (12.97°N) radiation curve.
*   **Battery Chemistry:** 10kWh static capacity mapping a 92% round-trip efficiency limit. Capped at 3kW simultaneous charge/discharge rates.
*   **Scoring Logic:** Scores are meticulously normalized to a strict `[0.0, 1.0]` limit against a theoretical naive baseline proxy, preventing runaway mathematical rewards.

---

## 🚀 Setup & Execution

### 1. Local Testing
```bash
# Clone the repository
git clone https://github.com/ArinHarwani/SmartGrid-Optima-.git
cd SmartGrid-Optima-

# Install the dependencies
pip install -e .

# Boot the OpenEnv background server
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### 2. Run LLM Inference
By default, the benchmark natively maps to `gpt-4.1-mini`.
```bash
export HF_TOKEN="your_hugging_face_token_here"
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4.1-mini"

python inference.py
```

---

## 📊 Standard Expected Baseline Scores

Using a zero-shot architecture (`gpt-4.1-mini` at `seed=42`), expected mathematical approximations map to:

| Task | Difficulty | Normalized Agent Output |
|------|-----------|----------------|
| `residential_summer_basic` | Easy | **~0.55** |
| `commercial_tod_optimization` | Medium | **~0.52** |
| `commercial_monsoon_resilience` | Hard | **~0.48** |

---

<p align="center">
   <i>Powered by the OpenEnv Core Framework | Designed for robust multi-agent scalability tests.</i>
</p>
