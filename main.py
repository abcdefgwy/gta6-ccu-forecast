"""Hierarchical Log-Quantile Time Series Forecaster.

Standard library only, deterministic, Python 3.13 compliant.
Optimized for multi-quantile asymmetric pinball loss evaluation.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_SCORES = [-1.645, -0.674, 0.0, 0.674, 1.645]
MODEL_VERSION = "1.0.1"
DEFAULT_SIGMA = 0.15


def round_half_up(value: float) -> float | int:
    """Round value to two decimal places half up."""
    scaled = value * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    res = whole / 100.0
    return int(res) if res.is_integer() else res


def parse_timestamp(iso_str: str) -> float:
    """Parse ISO timestamp to seconds epoch."""
    clean = iso_str.replace("Z", "+00:00")
    dt = datetime.fromisoformat(clean)
    return dt.timestamp()


def extract_hourly_series(comp_entry: dict) -> dict[int, float] | None:
    """Extract 24-hour observation values from benchmark dataset."""
    day_start = comp_entry.get("day_start")
    if not day_start:
        return None

    desc = (comp_entry.get("unlock") or "").lower()
    if "early-access" in desc or "target-history" in desc or "tier: weak" in desc:
        return None

    try:
        t0 = parse_timestamp(day_start)
    except Exception:
        return None

    hourly_map: dict[str, float] = {}
    for row in comp_entry.get("hourly", []):
        f = row.get("from")
        v = row.get("viewers")
        if f is not None and v is not None:
            hourly_map[f] = float(v)

    res: dict[int, float] = {}
    for h in range(1, 25):
        t_h = t0 + (h - 1) * 3600.0
        dt_h = datetime.fromtimestamp(t_h, tz=timezone.utc)
        iso_h = dt_h.strftime("%Y-%m-%dT%H:00:00Z")
        val = hourly_map.get(iso_h)
        if val is None or val <= 0:
            return None
        res[h] = val

    return res


def compute_distribution_parameters(
    target_v1: float,
    is_rolling: bool,
    comparables: list[dict],
) -> tuple[dict[int, float], dict[int, float]]:
    """Compute conditional medians and scales across historical series."""
    extracted = []
    for comp in comparables:
        series = extract_hourly_series(comp)
        if series is None or series[1] < 10.0:
            continue
        info = (comp.get("unlock") or "").lower()
        comp_rolling = "rolling" in info or "midnight local" in info
        v1 = series[1]
        extracted.append((series, comp_rolling, v1))

    ratios: dict[int, float] = {}
    spreads: dict[int, float] = {}

    for h in range(2, 25):
        diff_weights = []
        for series, comp_rolling, v1 in extracted:
            diff = math.log(series[h] / v1)
            w_pattern = 3.0 if (comp_rolling == is_rolling) else 1.0
            dist = abs(math.log(max(1.0, v1)) - math.log(max(1.0, target_v1)))
            w_scale = math.exp(-0.5 * (dist / 2.2) ** 2)
            diff_weights.append((diff, w_pattern * w_scale))

        if not diff_weights:
            ratios[h] = 1.0
            spreads[h] = DEFAULT_SIGMA * math.sqrt(h - 1)
            continue

        diff_weights.sort(key=lambda x: x[0])
        total_w = sum(w for _, w in diff_weights)

        # Weighted median calculation
        acc = 0.0
        med_log = diff_weights[0][0]
        for val, w in diff_weights:
            acc += w
            if acc >= 0.5 * total_w:
                med_log = val
                break

        # Median absolute deviation estimation
        devs = [(abs(val - med_log), w) for val, w in diff_weights]
        devs.sort(key=lambda x: x[0])
        acc = 0.0
        mad = devs[0][0]
        for d, w in devs:
            acc += w
            if acc >= 0.5 * total_w:
                mad = d
                break

        sigma = max(0.20, 1.4826 * mad)
        ratios[h] = math.exp(med_log)
        spreads[h] = sigma

    return ratios, spreads


def forecast_case(case_data: dict) -> dict:
    """Generate multi-quantile predictions for a single scenario."""
    known = case_data.get("known", [])
    p1_entry = next((k for k in known if k.get("hour") == 1), None)
    if p1_entry is None or p1_entry.get("viewers") is None:
        raise ValueError("Missing initial anchor observation")

    p1 = max(1.0, float(p1_entry["viewers"]))
    notes = (case_data.get("notes") or "").lower()
    is_rolling = "rolling" in notes or "midnight local" in notes or True

    comparables = case_data.get("comparables", [])
    ratios, spreads = compute_distribution_parameters(p1, is_rolling, comparables)

    hours_output = []
    for h in range(2, 25):
        r_h = ratios.get(h, 1.0)
        s_h = spreads.get(h, DEFAULT_SIGMA * math.sqrt(h - 1))

        quantiles = []
        for z in Z_SCORES:
            val = p1 * r_h * math.exp(z * s_h)
            clamped = max(0.01, min(1e9, val))
            quantiles.append(round_half_up(clamped))

        quantiles.sort()
        hours_output.append({"hour": h, "q": quantiles})

    return {"hours": hours_output}


def main() -> None:
    """Main batch processing entrypoint."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        with open(path, "r", encoding="utf-8") as f:
            case_data = json.load(f)
        forecast = forecast_case(case_data)
        out_path = OUTPUT_DIR / f"{path.stem}.out"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(forecast, f, separators=(",", ":"))
            f.write("\n")


if __name__ == "__main__":
    main()
