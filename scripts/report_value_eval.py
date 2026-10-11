"""Render artifacts/value_model/<version>/metrics.json as Markdown tables (pasted into docs/value/EVALUATION.md).

    uv run python scripts/report_value_eval.py v1
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = [("bayes", "Bayesian hierarchical hedonic"), ("lgbm_quantile", "LightGBM quantile (approx. draws)"),
          ("ols_with_sap", "OLS with SAP"), ("ols_without_sap", "OLS without SAP"),
          ("naive_la_type_median", "Naive: LA x type median")]


def main() -> None:
    v = sys.argv[1] if len(sys.argv) > 1 else "v1"
    m = json.loads((ROOT / "artifacts" / "value_model" / v / "metrics.json").read_text())
    lines = []
    for split, title in (("test", "Held-out test (unseen properties, 2008-2023)"),
                         ("out_of_time", "Out-of-time (unseen properties, 2024)")):
        n = m["bayes"][split]["n"]
        lines.append(f"### {title}: n = {n:,}\n")
        lines.append("| Model | MSE (log) | RMSE (log) | MAE (log) | R² (log) | R² (£) | RMSE £ | MAE £ | MAPE | Median APE | Spearman |")
        lines.append("| --- " * 11 + "|")
        for k, name in MODELS:
            r = m[k][split]
            lg, p = r["log"], r["price_gbp"]
            lines.append(f"| {name} | {lg['mse']:.4f} | {lg['rmse']:.4f} | {lg['mae']:.4f} | {lg['r2']:.3f} | "
                         f"{p['r2']:.3f} | {p['rmse']:,.0f} | {p['mae']:,.0f} | {p['mape']:.1%} | "
                         f"{p['median_ape']:.1%} | {p['spearman']:.3f} |")
        lines.append("")
        lines.append("| Model | CRPS (log) ↓ | Cov 50% | Cov 80% | Cov 90% | Cov 95% | Width 90% (log) | Pinball ↓ | PIT KS ↓ | Log score ↑ |")
        lines.append("| --- " * 10 + "|")
        for k, name in MODELS:
            d = m[k][split]["distribution"]
            iv = d["intervals"]
            lines.append(f"| {name} | {d['crps_log']:.4f} | {iv['50']['coverage']:.3f} | {iv['80']['coverage']:.3f} | "
                         f"{iv['90']['coverage']:.3f} | {iv['95']['coverage']:.3f} | {iv['90']['mean_width_log']:.3f} | "
                         f"{d['pinball_mean_5_50_95']:.4f} | {d['pit_ks_distance']:.3f} | {d['log_score_gaussian']:.3f} |")
        lines.append("")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data" / "interim" / f"value_eval_tables_{v}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
