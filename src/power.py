"""Sample size and MDE for a two-sample proportion test (equal arms)."""

from __future__ import annotations

import math

from scipy.stats import norm


def z_alpha_beta(alpha: float = 0.05, power: float = 0.80) -> tuple[float, float]:
    z_a = float(norm.ppf(1 - alpha / 2))
    z_b = float(norm.ppf(power))
    return z_a, z_b


def n_per_arm_proportion(
    p_baseline: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Users per arm to detect an absolute MDE on a binary metric."""
    if mde <= 0:
        raise ValueError("mde must be positive")
    p2 = min(max(p_baseline - mde, 1e-6), 1 - 1e-6)
    z_a, z_b = z_alpha_beta(alpha, power)
    var = p_baseline * (1 - p_baseline) + p2 * (1 - p2)
    n = (z_a + z_b) ** 2 * var / (mde**2)
    return int(math.ceil(n))


def mde_proportion(
    p_baseline: float,
    n_per_arm: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Absolute MDE (percentage points if you multiply by 100) given n per arm."""
    if n_per_arm <= 0:
        return float("nan")
    z_a, z_b = z_alpha_beta(alpha, power)
    return (z_a + z_b) * math.sqrt(2 * p_baseline * (1 - p_baseline) / n_per_arm)


def days_to_power(
    n_per_arm_needed: int,
    unique_users_per_day: float,
) -> float:
    """Calendar days if traffic is split 50/50 across arms."""
    if unique_users_per_day <= 0:
        return float("inf")
    daily_per_arm = unique_users_per_day / 2
    return n_per_arm_needed / daily_per_arm
