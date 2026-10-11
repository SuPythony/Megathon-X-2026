"""Build the demo city: many homes' digital twins in batch, plus the business insights for the Lloyds dashboard.

    uv run python scripts/build_demo_city.py                 # 60 homes -> artifacts/demo_city/v1/
    uv run python scripts/build_demo_city.py --n 60 --samples 60

Homes are SYNTHETIC over real Manchester postcode districts (greentwin/twin/city_assumptions.json) until an EPC
extract is available. Numbers come from the stub physics until greentwin/twin/physics.py exists (engine field says
which). See docs/business_insight.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from greentwin.twin.city import build_city, summary, write_city  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)        # laptop-friendly: P2's half-hourly PV layer is memory-heavy
    ap.add_argument("--samples", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "demo_city" / "v1"))
    args = ap.parse_args()
    t0 = time.time()
    homes, areas, config = build_city(args.n, args.samples, args.seed)
    config["build_seconds"] = round(time.time() - t0, 1)
    config["summary"] = summary(homes, areas)
    for p in write_city(Path(args.out), homes, areas, config):
        print(f"wrote {p} ({p.stat().st_size / 1024:.0f} KB)")
    print(json.dumps(config["summary"], indent=2))
    print(f"built in {config['build_seconds']} s")


if __name__ == "__main__":
    main()
