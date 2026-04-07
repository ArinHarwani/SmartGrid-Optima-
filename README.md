# SmartGrid-Optima ⚡🔋☀️

**A Smart Energy Management RL Environment for Bangalore, India**

Built with the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework for the Meta PyTorch Hackathon.

---

## 🎯 What is SmartGrid-Optima?

SmartGrid-Optima simulates a real-world energy management scenario where an AI agent manages:

- **Solar Panels** (5kW peak) — Free energy from sunlight
- **Battery Storage** (10kWh, 92% efficiency) — Store cheap energy for expensive hours
- **Grid Connection** (BESCOM pricing) — Buy/sell electricity at market rates
- **Home Load** — Residential or commercial consumption profiles

The agent's goal is to **minimize electricity cost** over a 24-hour period by choosing the optimal action each hour.

## 🌍 Why This Matters

India's electricity grid faces massive peak-hour demand. Smart energy management can:
- Reduce household electricity bills by 30-50%
- Optimize renewable energy usage
- Build grid resilience during outages (monsoon season)
- Support India's net-zero goals through efficient battery utilization

---

## 🎮 Action Space

| Action | Name | Description |
|--------|------|-------------|
| `0` | **Idle** | Let solar handle load; buy from grid if needed |
| `1` | **Charge** | Charge battery from solar (free) or grid (paid) |
| `2` | **Discharge** | Use battery power to avoid expensive grid purchases |
| `3` | **Sell** | Sell excess solar/battery power to grid for revenue |

## 👁️ Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `hour` | int | Current hour (0-23) |
| `solar_output_kw` | float | Solar panel output in kW |
| `cloud_cover_pct` | float | Cloud coverage (0-100%) |
| `battery_soc` | float | Battery state of charge (0.0-1.0) |
| `battery_kwh` | float | Battery energy in kWh (0-10) |
| `grid_price_buy` | float | Grid electricity buy price (₹/kWh) |
| `grid_price_sell` | float | Grid electricity sell price (₹/kWh) |
| `grid_available` | bool | Is grid power available |
| `home_load_kw` | float | Home power consumption in kW |
| `cost_cumulative` | float | Total cost so far (₹) |
| `task_id` | str | Active task identifier |
| `persona` | str | "residential" or "commercial" |

---

## 📋 Tasks (Easy → Medium → Hard)

### Task 1: `residential_summer_basic` (EASY)
- **Persona:** Residential (flat ₹5.90/kWh)
- **Weather:** Clear sky (0-10% clouds)
- **Battery:** Starts at 30-80% SoC
- **Challenge:** Optimize solar usage on a sunny day

### Task 2: `commercial_tod_optimization` (MEDIUM)
- **Persona:** Commercial (Time-of-Day pricing: ₹8-9.50/kWh)
- **Weather:** Partly cloudy (20-40%)
- **Battery:** Starts at 20-50% SoC
- **Challenge:** Exploit peak/off-peak price arbitrage

### Task 3: `commercial_monsoon_resilience` (HARD)
- **Persona:** Commercial (ToD pricing)
- **Weather:** Heavy monsoon (70-95% clouds)
- **Grid:** Random outages (15% chance per hour)
- **Battery:** Starts at 10-30% SoC
- **Challenge:** Maintain power during blackouts while minimizing cost

---

## 💰 Reward Function

**Raw Reward:** `Cost_NoAI - Cost_WithAI` (savings compared to naive baseline)

**Penalties:**
- **Blackout:** -100 (SoC drops below 10% during grid outage)
- **Excessive Cycling:** -2 (rapidly alternating charge/discharge)

**Normalization:** `clamp((Raw - MinReward) / (MaxReward - MinReward), 0.0, 1.0)`

---

## 🏗️ Architecture

```
smartgrid_optima/
├── models.py              ← Pydantic types: EnergyAction, EnergyObservation, EnergyState
├── data.py                ← Offline: BESCOM pricing, solar curves, weather, load profiles
├── graders.py             ← 3 task graders scoring 0.0-1.0
├── client.py              ← WebSocket client (SmartGridEnv)
├── inference.py           ← LLM agent using OpenAI Client
├── openenv.yaml           ← OpenEnv manifest
├── Dockerfile             ← Container definition
├── pyproject.toml         ← Dependencies
└── server/
    ├── smartgrid_environment.py  ← Core 24-hour simulation
    ├── app.py                     ← FastAPI server
    └── requirements.txt           ← Server dependencies
```

---

## ⚡ Hardware & Physics

| Component | Specification |
|-----------|---------------|
| Solar System | 5kW peak, Bangalore bell curve (12.97°N) |
| Battery | 10kWh capacity, 92% round-trip efficiency |
| Max Charge Rate | 3kW per hour |
| Max Discharge Rate | 3kW per hour |
| Min SoC | 10% (safety reserve) |

### BESCOM Pricing (Bangalore 2025-26)

| Persona | Period | Rate (₹/kWh) |
|---------|--------|---------------|
| Residential | All hours | ₹5.90 (buy) / ₹3.86 (sell) |
| Commercial | Normal (06-18h) | ₹8.00 |
| Commercial | Peak (18-22h) | ₹9.50 |
| Commercial | Night (22-06h) | ₹7.00 |
| Commercial | Sell | ₹3.20 |

---

## 🚀 Setup & Usage

### Local Development

```bash
# Clone the repository
git clone <your-repo-url>
cd smartgrid_optima

# Install dependencies
pip install -e .

# Run the server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Test the environment
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" \
  -d '{"task_id": "residential_summer_basic", "seed": 42}'
```

### Docker

```bash
docker build -t smartgrid-optima .
docker run -p 8000:8000 smartgrid-optima
```

### Run Inference

```bash
export HF_TOKEN="your-token"
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4.1-mini"
python inference.py
```

---

## 📊 Baseline Scores

| Task | Difficulty | Baseline Score |
|------|-----------|----------------|
| `residential_summer_basic` | Easy | ~0.55 |
| `commercial_tod_optimization` | Medium | ~0.52 |
| `commercial_monsoon_resilience` | Hard | ~0.48 |

*Scores using GPT-4.1-mini with seed=42*

---

## 📝 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_BASE_URL` | LLM API endpoint | `https://api.openai.com/v1` |
| `MODEL_NAME` | Model identifier | `gpt-4.1-mini` |
| `HF_TOKEN` | API key | *(required, no default)* |

---

## 🔗 Links

- [OpenEnv Framework](https://github.com/meta-pytorch/OpenEnv)
- [OpenEnv Course](https://github.com/raun/openenv-course)
- [BESCOM Tariff Orders](https://bescom.karnataka.gov.in/)

---

*Built for the Meta PyTorch OpenEnv Hackathon 2026*
