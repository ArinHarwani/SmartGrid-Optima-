"""
Offline data module for SmartGrid-Optima.

Contains all physics, pricing, weather, and load data for Bangalore 2025-26.
100% offline — no external API calls. Uses local dictionaries.

Data Sources:
    - BESCOM/KERC tariff orders 2025-26
    - Bangalore solar irradiance data (12.97°N latitude)
    - Typical Bangalore weather patterns
"""

import math
import random
from typing import Dict, List, Tuple

# =============================================================================
# 1. BESCOM/KERC PRICING (Bangalore 2025-26)
# =============================================================================

PRICING = {
    "residential": {
        "name": "Residential (LT-1)",
        "buy_rate": 5.90,        # Flat ₹5.90/kWh
        "sell_rate": 3.86,       # Net metering ₹3.86/kWh
        "is_tod": False,         # No time-of-day pricing
    },
    "commercial": {
        "name": "Commercial (HT/LT-3 ToD)",
        "sell_rate": 3.20,       # ₹3.20/kWh sell rate
        "is_tod": True,          # Time-of-day pricing
        "tod_rates": {
            "peak":   {"rate": 9.50, "hours": list(range(18, 22))},     # 18:00-22:00
            "night":  {"rate": 7.00, "hours": list(range(22, 24)) + list(range(0, 6))},  # 22:00-06:00
            "normal": {"rate": 8.00, "hours": list(range(6, 18))},      # 06:00-18:00
        },
    },
}


def get_grid_buy_price(persona: str, hour: int) -> float:
    """Get grid buy price for given persona and hour."""
    config = PRICING[persona]
    if not config["is_tod"]:
        return config["buy_rate"]

    for period_name, period in config["tod_rates"].items():
        if hour in period["hours"]:
            return period["rate"]
    return config["tod_rates"]["normal"]["rate"]


def get_grid_sell_price(persona: str) -> float:
    """Get grid sell price for given persona."""
    return PRICING[persona]["sell_rate"]


# =============================================================================
# 2. SOLAR GENERATION (5kW Peak, Bangalore Bell Curve)
# =============================================================================

# Bangalore: ~12.97°N latitude
# Sunrise ~6:00, Sunset ~18:30 (average)
# Peak solar at ~12:30 PM

SOLAR_PEAK_KW = 5.0          # 5kW peak system
SOLAR_SUNRISE_HOUR = 6.0     # Sunrise
SOLAR_SUNSET_HOUR = 18.5     # Sunset
SOLAR_PEAK_HOUR = 12.25      # Peak generation time


def get_solar_max_kw(hour: int) -> float:
    """
    Get maximum possible solar output for a given hour using a bell curve.
    Models a 5kW peak system in Bangalore.
    """
    if hour < SOLAR_SUNRISE_HOUR or hour > SOLAR_SUNSET_HOUR:
        return 0.0

    # Gaussian bell curve centered at peak hour
    sigma = 2.5  # Spread of the curve
    solar = SOLAR_PEAK_KW * math.exp(
        -0.5 * ((hour - SOLAR_PEAK_HOUR) / sigma) ** 2
    )
    return round(max(0.0, solar), 3)


def get_solar_actual_kw(hour: int, cloud_cover_pct: float) -> float:
    """
    Get actual solar output factoring in cloud coverage.
    Formula: Solar_Actual = Solar_Max × (1 - CloudPercentage/100)
    """
    solar_max = get_solar_max_kw(hour)
    actual = solar_max * (1.0 - cloud_cover_pct / 100.0)
    return round(max(0.0, actual), 3)


# =============================================================================
# 3. BANGALORE WEATHER PROFILES (Offline)
# =============================================================================

# Hourly cloud coverage percentages for typical Bangalore days
# Each profile is a list of 24 values (one per hour, 0-23)

WEATHER_PROFILES: Dict[str, List[float]] = {
    "clear_summer": [
        0, 0, 0, 0, 0, 2, 5, 5, 3, 2, 2, 3, 5, 5, 3, 5, 8, 10, 5, 3, 2, 0, 0, 0
    ],
    "partly_cloudy": [
        5, 5, 3, 5, 8, 10, 15, 20, 25, 30, 25, 20, 25, 30, 35, 30, 25, 20, 15, 10, 8, 5, 5, 5
    ],
    "monsoon_heavy": [
        60, 55, 50, 55, 60, 70, 75, 80, 85, 90, 95, 90, 85, 80, 85, 90, 95, 90, 85, 80, 75, 70, 65, 60
    ],
    "monsoon_moderate": [
        40, 35, 30, 35, 40, 50, 60, 65, 70, 75, 80, 75, 70, 65, 70, 75, 80, 75, 70, 60, 55, 50, 45, 40
    ],
    "winter_clear": [
        5, 5, 5, 5, 8, 10, 8, 5, 3, 2, 2, 3, 5, 5, 8, 10, 12, 10, 8, 5, 5, 5, 5, 5
    ],
    "pre_monsoon_buildup": [
        10, 8, 5, 8, 15, 20, 25, 30, 35, 40, 50, 55, 60, 65, 70, 65, 60, 55, 50, 40, 30, 20, 15, 10
    ],
}


def get_cloud_cover(hour: int, base_cloud_min: float, base_cloud_max: float, seed: int = None) -> float:
    """
    Generate cloud coverage for a specific hour with controlled randomness.
    Uses base range from task configuration + profile-based variation.
    """
    if seed is not None:
        rng = random.Random(seed + hour)
    else:
        rng = random.Random()

    base = rng.uniform(base_cloud_min, base_cloud_max)

    # Add slight hourly variation (±5%)
    variation = rng.uniform(-5, 5)
    cloud = max(0.0, min(100.0, base + variation))
    return round(cloud, 1)


# =============================================================================
# 4. HOME LOAD PROFILES (kW per hour)
# =============================================================================

# Typical hourly power consumption for residential and commercial buildings

LOAD_PROFILES: Dict[str, List[float]] = {
    "residential": [
        # 0    1    2    3    4    5    6    7    8    9   10   11
        0.3, 0.2, 0.2, 0.2, 0.2, 0.3, 0.8, 1.5, 1.2, 0.8, 0.6, 0.5,
        # 12  13   14   15   16   17   18   19   20   21   22   23
        0.7, 0.5, 0.4, 0.5, 0.6, 1.0, 2.0, 2.5, 2.2, 1.8, 1.0, 0.5,
    ],
    "commercial": [
        # 0    1    2    3    4    5    6    7    8    9   10   11
        0.5, 0.4, 0.3, 0.3, 0.3, 0.5, 1.0, 2.0, 3.5, 4.0, 4.2, 4.0,
        # 12  13   14   15   16   17   18   19   20   21   22   23
        3.8, 4.0, 4.2, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.8, 0.5,
    ],
}


def get_home_load(persona: str, hour: int, noise_factor: float = 0.1, seed: int = None) -> float:
    """
    Get home load for given persona and hour with slight random variation.
    """
    if seed is not None:
        rng = random.Random(seed + hour + 1000)
    else:
        rng = random.Random()

    base_load = LOAD_PROFILES[persona][hour]
    noise = rng.uniform(-noise_factor, noise_factor) * base_load
    return round(max(0.1, base_load + noise), 3)


# =============================================================================
# 5. BATTERY SPECIFICATIONS
# =============================================================================

BATTERY = {
    "capacity_kwh": 10.0,          # 10 kWh total capacity
    "charge_efficiency": 0.96,     # 96% charge efficiency
    "discharge_efficiency": 0.96,  # 96% discharge efficiency
    "roundtrip_efficiency": 0.92,  # 92% round-trip (96% × 96%)
    "max_charge_rate_kw": 3.0,     # Max 3kW charge rate per hour
    "max_discharge_rate_kw": 3.0,  # Max 3kW discharge rate per hour
    "min_soc": 0.10,               # 10% minimum state of charge (safety)
    "max_soc": 1.00,               # 100% maximum
}


# =============================================================================
# 6. TASK CONFIGURATIONS
# =============================================================================

TASKS = {
    "residential_summer_basic": {
        "difficulty": "easy",
        "persona": "residential",
        "cloud_min": 0.0,
        "cloud_max": 10.0,
        "soc_min": 0.30,
        "soc_max": 0.80,
        "grid_outage_chance": 0.0,
        "description": "Sunny summer day with residential flat-rate pricing. "
                       "Optimize battery usage to minimize grid purchases.",
    },
    "commercial_tod_optimization": {
        "difficulty": "medium",
        "persona": "commercial",
        "cloud_min": 20.0,
        "cloud_max": 40.0,
        "soc_min": 0.20,
        "soc_max": 0.50,
        "grid_outage_chance": 0.0,
        "description": "Partly cloudy day with commercial ToD pricing. "
                       "Exploit peak/off-peak price differences.",
    },
    "commercial_monsoon_resilience": {
        "difficulty": "hard",
        "persona": "commercial",
        "cloud_min": 70.0,
        "cloud_max": 95.0,
        "soc_min": 0.10,
        "soc_max": 0.30,
        "grid_outage_chance": 0.15,
        "description": "Monsoon day with heavy clouds, commercial ToD pricing, "
                       "and random grid outages. Maintain operations and minimize cost.",
    },
}


# =============================================================================
# 7. REWARD CONSTANTS
# =============================================================================

REWARD = {
    "blackout_penalty": -100.0,     # Penalty for blackout (SoC < min during outage)
    "cycling_penalty": -2.0,        # Penalty per excessive battery cycle
    "max_reward": 50.0,             # Theoretical max savings (perfect arbitrage)
    "min_reward": -150.0,           # Theoretical worst (blackout + cycling)
}


def normalize_reward(raw_reward: float) -> float:
    """
    Normalize raw reward to 0.0-1.0 range.
    Formula: clamp((raw - min) / (max - min), 0.0, 1.0)
    """
    max_r = REWARD["max_reward"]
    min_r = REWARD["min_reward"]
    if max_r == min_r:
        return 0.5
    normalized = (raw_reward - min_r) / (max_r - min_r)
    return round(max(0.0, min(1.0, normalized)), 4)
