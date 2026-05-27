"""Nebius preset → USD/min price table + cost-estimation helpers.

Prices are hardcoded approximations; **verify against the current Nebius pricing
console** (https://nebius.com/services/serverless-jobs/pricing) before quoting
numbers anywhere user-facing. The talk-prep PR is the place to refresh these.

Lookup key: ``(platform, preset, preemptible)``. Preemptible prices apply only
when the job actually ran on preemptible capacity (~70-85% discount vs on-demand
on Nebius today). On-demand cost for the same wall time is computed via
``on_demand_estimate()`` (used in savings rollups).
"""

from __future__ import annotations

# TODO(verify): refresh from Nebius pricing console before quoting cost numbers.
# Last manual check: 2026-05 — values are *order-of-magnitude* placeholders.
PRICE_PER_MIN_USD: dict[tuple[str, str, bool], float] = {
    # CPU
    ("cpu-e2", "4vcpu-16gb",  False): 0.003,
    ("cpu-e2", "8vcpu-32gb",  False): 0.006,
    # GPU L40S — preemptible vs on-demand
    ("gpu-l40s-d", "1gpu-16vcpu-96gb", True):  0.012,
    ("gpu-l40s-d", "1gpu-16vcpu-96gb", False): 0.060,
    # GPU H100/H200 SXM
    ("gpu-h100-sxm", "1gpu-16vcpu-200gb", True):  0.040,
    ("gpu-h100-sxm", "1gpu-16vcpu-200gb", False): 0.180,
    ("gpu-h200-sxm", "1gpu-16vcpu-200gb", True):  0.060,
    ("gpu-h200-sxm", "1gpu-16vcpu-200gb", False): 0.250,
}


def _lookup(platform: str, preset: str, preemptible: bool) -> float:
    rate = PRICE_PER_MIN_USD.get((platform, preset, preemptible))
    if rate is None:
        # Unknown combo → return 0 rather than blow up; surfaces as "n/a" in summary.
        return 0.0
    return rate


def estimate_cost(platform: str, preset: str, preemptible: bool, run_s: float) -> float:
    """USD estimate for a job that ran ``run_s`` seconds on the given machine."""
    if run_s <= 0:
        return 0.0
    return _lookup(platform, preset, preemptible) * (run_s / 60.0)


def on_demand_estimate(platform: str, preset: str, run_s: float) -> float:
    """What the same wall time would have cost at on-demand rates.

    Used by ``write_run_summary`` to quantify the savings from running on
    preemptible capacity.
    """
    if run_s <= 0:
        return 0.0
    return _lookup(platform, preset, preemptible=False) * (run_s / 60.0)
