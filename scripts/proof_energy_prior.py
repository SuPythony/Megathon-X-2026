"""Collect verifiable evidence for the M04 report: data provenance, split balance, health check, tests, commits.

    uv run python scripts/proof_energy_prior.py   ->  artifacts/m04_energy_prior/v2/proof.json
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/m04_energy_prior/v2"
RAW = ROOT / "data/raw/need"


def sh(*cmd: str) -> str:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True).stdout


def sha256(path: Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    out = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    # Data provenance: what was downloaded, from where, how big, checksums.
    files = []
    for name, url in (("anon2025_4million.zip", "https://assets.publishing.service.gov.uk/media/685c04734cd6b03168709827/anon2025_4million.zip"),
                      ("anon2025_4million.csv", None),
                      ("NEED-2025-metadata.ods", "https://assets.publishing.service.gov.uk/media/685c02aa0433072fce0e1025/NEED-2025-anonymised-dataset-metadata.ods")):
        p = RAW / name
        if p.exists():
            files.append({"file": name, "bytes": p.stat().st_size, "sha256": sha256(p), "url": url})
    table = ROOT / "data/processed/need_intensity.parquet"
    df = pd.read_parquet(table)
    out["data"] = {
        "publisher": "DESNZ, National Energy Efficiency Data-Framework (NEED) anonymised data 2025",
        "page": "https://www.gov.uk/government/statistics/national-energy-efficiency-data-framework-need-anonymised-data-2025",
        "licence": "Open Government Licence v3", "files": files,
        "rows": int(len(df)), "unique_properties": int(df.property_id.nunique()),
        "gas_heated": int((df.main_heat_fuel == "gas").sum()),
        "valid_gas_target": int(((df.main_heat_fuel == "gas") & df.gas_kwh.notna()).sum()),
        "valid_elec_target": int(df.elec_kwh.notna().sum()),
    }
    # Split: shares and worst stratum-level imbalance.
    shares = df.split.value_counts(normalize=True).round(5).to_dict()
    gaps = {}
    for col in ("prop_type", "age_band", "floor_band", "region", "main_heat_fuel"):
        t = pd.crosstab(df[col], df.split, normalize="columns") * 100
        gaps[col] = round(float((t.max(axis=1) - t.min(axis=1)).max()), 4)
    out["split"] = {"shares": shares, "counts": df.split.value_counts().to_dict(),
                    "max_gap_pp_by_stratum": gaps, "disjoint": bool(df.property_id.is_unique)}
    # Health check and tests, run now.
    chk = subprocess.run([sys.executable, "scripts/check_energy_prior.py", "--json"], cwd=ROOT,
                         capture_output=True, text=True)
    out["health_check"] = json.loads(chk.stdout)
    pt = sh(sys.executable, "-m", "pytest", "-q")
    m = re.search(r"(\d+) passed(?:, (\d+) skipped)?(?:, (\d+) failed)?.* in ([\d.]+)s", pt)
    out["tests"] = {"passed": int(m.group(1)) if m else None, "skipped": int(m.group(2) or 0) if m else None,
                    "failed": int(m.group(3) or 0) if m else None, "summary": pt.strip().splitlines()[-1] if pt else ""}
    out["tests"]["files"] = sorted(p.name for p in (ROOT / "tests").glob("test_*.py"))
    # Tuning rounds: EBM before and after the wide search.
    r1 = json.loads((RUN / "metrics_round1_ebm_base.json").read_text())
    r2 = json.loads((RUN / "metrics.json").read_text())
    out["ebm_rounds"] = {f: {"round1": {"r2": r1[f]["test"]["ebm"]["r2"], "mae": r1[f]["test"]["ebm"]["mae"],
                                        "params": r1[f]["best_params"]["ebm"]},
                             "round2": {"r2": r2[f]["test"]["ebm"]["r2"], "mae": r2[f]["test"]["ebm"]["mae"],
                                        "params": r2[f]["best_params"]["ebm"]}}
                         for f in ("electricity", "gas")}
    out["commits"] = [line.split(" ", 1) for line in sh("git", "log", "--format=%h %s", "main..HEAD").splitlines()]
    out["branch"] = sh("git", "rev-parse", "--abbrev-ref", "HEAD").strip()
    out["model_sha256"] = sha256(RUN / "model.pkl")
    out["reproduce"] = [
        "uv sync",
        "curl -L -o data/raw/need/anon2025_4million.zip <NEED 4M URL>  &&  unzip it in data/raw/need/",
        "uv run python -m greentwin.data.need",
        "uv run python scripts/train_energy_prior.py --trials-lgb 20 --trials-xgb 12 --trials-ebm 10",
        "uv run python scripts/train_energy_prior.py --fixed-params artifacts/m04_energy_prior/v2/config_round1.json --retune ebm --ebm-space wide --trials-ebm 8",
        "uv run python scripts/bayes_energy_prior.py",
        "uv run python scripts/diagnostics_energy_prior.py",
        "uv run python scripts/proof_energy_prior.py",
        "uv run python scripts/report_energy_prior.py && uv run python scripts/model_card_energy_prior.py",
        "uv run python scripts/check_energy_prior.py && uv run pytest -q",
    ]
    (RUN / "proof.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {RUN / 'proof.json'}: health {out['health_check']['ok']}, tests {out['tests']['summary']}")


if __name__ == "__main__":
    main()
