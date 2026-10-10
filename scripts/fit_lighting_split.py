"""Learn how much of smart_energy_management should be lighting (decision D-26).

The brief fixes seven components, so lighting cannot be an eighth. It joins smart_energy_management as
'controls and efficient electricity use'. The split is learned: the same simplex NNLS as the weights, with
low-energy lighting % as its own attribute, on the RdSAP 10 era; split = w_light / (w_light + w_controls),
averaged over the four modelled targets.

    uv run python scripts/fit_lighting_split.py --epc-dir data/interim/epc25
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from train_score_weights import ATTR, TARGETS, load_epc  # noqa: E402

from greentwin.score import learn as L  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epc-dir", default="data/interim/epc25")
    ap.add_argument("--era", default="rdsap10")
    ap.add_argument("--boot", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    f = load_epc(a.epc_dir, None, a.era).dropna(subset=["low_energy_lighting_pct"])
    f = f.assign(s_lighting=f["low_energy_lighting_pct"].clip(0, 100))
    tr, te = f[f.split == "train"], f[f.split == "test"]
    X = ATTR + ["s_lighting"]
    res, splits = {}, []
    rng = np.random.default_rng(a.seed)
    for t in TARGETS:
        y = f"y_{t}"
        fit = L.simplex_nnls(tr[X], tr[y])
        w = fit.weights
        split = w["s_lighting"] / max(w["s_lighting"] + w["s_smart_energy_management"], 1e-9)
        base = L.simplex_nnls(tr[ATTR], tr[y])
        boot = []
        sub = tr.sample(min(200_000, len(tr)), random_state=a.seed)
        for _ in range(a.boot):
            i = rng.integers(0, len(sub), len(sub))
            wb = L.simplex_nnls(sub[X].iloc[i], sub[y].iloc[i]).weights
            boot.append(wb["s_lighting"] / max(wb["s_lighting"] + wb["s_smart_energy_management"], 1e-9))
        res[t] = {"weights_with_lighting": w, "lighting_share_of_smart": round(split, 4),
                  "lighting_share_boot_p5_p95": np.percentile(boot, [5, 95]).round(4).tolist(),
                  "r2_test_with_lighting": round(L.r2(te[y], fit.predict(te[X])), 4),
                  "r2_test_without_lighting": round(L.r2(te[y], base.predict(te[ATTR])), 4)}
        splits.append(split)
        print(t, json.dumps(res[t]))
    out = {"era": a.era, "n": len(f), "n_train": len(tr), "n_test": len(te), "targets": res,
           "lighting_share_of_smart_mean": round(float(np.mean(splits)), 4), "seed": a.seed}
    p = ROOT / "artifacts" / "score_lighting" / "v1"
    p.mkdir(parents=True, exist_ok=True)
    (p / "metrics.json").write_text(json.dumps(out, indent=2, default=float))
    print("mean lighting share of smart:", out["lighting_share_of_smart_mean"])


if __name__ == "__main__":
    main()
