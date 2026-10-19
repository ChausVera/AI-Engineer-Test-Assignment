"""Tool 3: assess a customer's flotation concentrate against the benchmark plant data.

Everything numeric is computed here from the CSV; the LLM only writes the answer from the returned dict.
Reuses the verified functions in analysis.py.

CLI test:  python assess_flotation.py 3.1 64.5
"""
import json
import sys
from pathlib import Path

import numpy as np

from analysis import SILICA, granger_table, lag_table, load

IRON = "% Iron Concentrate"
DEFAULT_CSV = "flotation_process_data_hourly.csv"

# Extra data to request, in priority order, each tied to a limit of the current data.
DATA_REQUESTS = [
    {
        "data": "Operator / control-system logs: timestamped amina and starch setpoint changes, "
                "manual vs automatic mode, who changed them and why",
        "why": "Tests whether dosing is really reactive. Our data only shows silica leading amina in time; "
               "logs show when dosing decisions were actually made.",
    },
    {
        "data": "Lab sampling times vs result reporting times for concentrate silica (and online "
                "analyser data, if any)",
        "why": "The 3-hour lead could partly be reporting delay. Separating sample time from report time "
               "shows how late the operators' information really is.",
    },
    {
        "data": "The customer's own hourly data for several months, with the same columns "
                "(feed and concentrate Fe/Si, reagent flows, pH, density, air flow, levels)",
        "why": "The benchmark is a different plant. Their data tells us whether their 3.1% Si / 64.5% Fe "
               "is normal for their feed and whether the same timing pattern exists.",
    },
    {
        "data": "Tailings grade (Fe, Si) and flows",
        "why": "Without tailings we cannot compute recovery or mass yield, so we cannot say whether "
               "a higher grade would cost recovery.",
    },
    {
        "data": "Ore characteristics: particle size (e.g. P80), mineralogy / liberation, slimes or clay content, "
                "feed rate",
        "why": "Feed silica alone does not explain concentrate quality. These are the usual confounders "
               "we could not control for.",
    },
    {
        "data": "Reagent and water details: amina and starch concentration and preparation, supplier "
                "batch, water quality and temperature",
        "why": "Dosing flow alone ignores reagent strength and conditions, which change what a given "
               "flow does.",
    },
    {
        "data": "Higher-frequency data (e.g. 20-second or 1-minute) and equipment events "
                "(downtime, maintenance)",
        "why": "Needed to look for a leading indicator that moves hours before silica does.",
    },
]

BASE_LIMITS = [
    "Benchmark data is from a different plant (Brazil, Mar-Sep 2017), so results are a reference point, "
    "not a diagnosis of the customer's plant.",
    "There is no tailings data, so recovery cannot be assessed; raising grade may cost recovery.",
]

TIMING_LIMITS = [
    "Timing relationships show predictive lead only, not cause. Feed quality and other process variables "
    "were not controlled for.",
    "The timing effect is small (about 1-2% of amina variation), so it is not a proven main cause "
    "of the customer's silica level.",
]

TIMING_LIMITS = [
    "Timing relationships show a predictive lead only, not cause.",
    "Feed quality and other process variables were not controlled for.",
    "The effect is small, so it is not a proven main cause of the customer's silica level.",
    "Confirming it needs the customer's operator logs.",
]


def _benchmark(df, si, fe):
    d = df[[IRON, SILICA]].dropna()
    slope, intercept = np.polyfit(d[IRON], d[SILICA], 1)
    resid = d[SILICA] - (slope * d[IRON] + intercept)
    expected_si = slope * fe + intercept
    similar = d[(d[IRON] - fe).abs() <= 0.5]
    return {
        "customer": {"silica": si, "iron": fe},
        "share_of_hours_with_lower_silica_pct": round(float((d[SILICA] < si).mean() * 100), 1),
        "share_of_hours_with_higher_iron_pct": round(float((d[IRON] > fe).mean() * 100), 1),
        "benchmark_median": {"silica": round(float(d[SILICA].median()), 2),
                             "iron": round(float(d[IRON].median()), 2)},
        "expected_silica_at_customer_iron": round(float(expected_si), 2),
        "silica_residual_z": round(float((si - expected_si) / resid.std()), 2),
        "hours_with_similar_iron": {
            "n": int(len(similar)),
            "median_silica": round(float(similar[SILICA].median()), 2) if len(similar) else None,
        },
    }


def _timing(df):
    lags = lag_table(df)
    r = {(direction, h): rr for direction, h, rr, *_ in lags}
    gr = granger_table(df)
    gains = {(lab, tr, p): g for lab, tr, p, _, _, g, _ in gr}
    ratios = {}
    for tr in ("levels", "differences"):
        for p in (3, 6):
            rev = gains[("amina -> silica", tr, p)]
            fwd = gains[("silica -> amina", tr, p)]
            ratios[f"{tr}_{p}lags"] = round(fwd / rev, 1) if rev > 0 else None
    vals = [v for v in ratios.values() if v is not None]
    if vals and len(vals) == 4 and all(v >= 2 for v in vals):
        verdict = "consistent_with_reactive_dosing"
    elif ratios["levels_3lags"] and ratios["levels_3lags"] >= 2 and (ratios["differences_3lags"] or 0) < 2:
        verdict = "possibly_shared_drift"
    else:
        verdict = "inconclusive"
    return {
        "corr_silica_leads_amina_3h": round(float(r[("silica -> amina", 3)]), 3),
        "corr_amina_leads_silica_3h": round(float(r[("amina -> silica", 3)]), 3),
        "amina_variation_explained_by_past_silica_pct": [
            round(100 * min(v for k, v in gains.items() if k[0] == "silica -> amina"), 1),
            round(100 * max(v for k, v in gains.items() if k[0] == "silica -> amina"), 1),
        ],
        "predictive_lead_ratio_silica_vs_amina": ratios,
        "verdict": verdict,
    }


def assess_flotation(silica_concentrate: float, iron_concentrate: float,
                     # deeper_research: bool = False,   # DISABLED: re-enable together with the block below
                     csv_path: str = DEFAULT_CSV) -> dict:
    """Assess a customer's concentrate quality using the benchmark flotation plant data.

    Compares the customer's silica and iron grades with the benchmark plant's hourly data
    (percentiles, expected silica at that iron grade, similar-iron hours).
    (Deeper research, i.e. the timing analysis and the extra-data requests, is disabled; see below.)
    Data quality checks live in analysis.py and are not part of this output.

    Args:
        silica_concentrate: customer's % silica in concentrate (e.g. 3.1).
        iron_concentrate: customer's % iron in concentrate (e.g. 64.5).
        csv_path: path to flotation_process_data_hourly.csv.
    """
    if not Path(csv_path).exists():
        return {"status": "data_unavailable", "message": "Flotation data file not found."}
    df = load(csv_path)
    out = {
        "status": "ok",
        "benchmark": _benchmark(df, silica_concentrate, iron_concentrate),
        "limitations": list(BASE_LIMITS),
    }
    # --- DISABLED: deeper research (timing analysis + extra data requests) ---
    # if deeper_research:
    #     out["timing"] = _timing(df)
    #     out["limitations"] += TIMING_LIMITS
    #     out["data_requests"] = DATA_REQUESTS
    # else:
    #     out["deeper_research_available"] = True
    return out


if __name__ == "__main__":
    # python assess_flotation.py [silica] [iron] [csv_path]
    si = float(sys.argv[1]) if len(sys.argv) > 1 else 3.1
    fe = float(sys.argv[2]) if len(sys.argv) > 2 else 64.5
    path = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_CSV
    print(json.dumps(assess_flotation(si, fe, csv_path=path), indent=2))