"""Dev twin server for the 3D viewer: re-simulates any retrofit package and appliance list on the showcase homes.

    uv run python scripts/twin_server.py      # serves http://127.0.0.1:8765

The viewer (web/twin-viewer, `npm run dev`) proxies /api to this server. Without it, the viewer falls back to the
precomputed single-retrofit scenarios in the scene JSON. Not the production API (that's greentwin/api, Part 5):
showcase homes only, no auth, localhost only.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from greentwin.twin.inputs import build_physics_inputs  # noqa: E402
from greentwin.twin.physics_api import get_run_physics  # noqa: E402
from greentwin.twin.retrofits import catalogue  # noqa: E402
from greentwin.twin.scenarios import run_scenario  # noqa: E402
from greentwin.twin.scene import AppliancePlacement, EnergyOverlay, ScenarioView  # noqa: E402
from greentwin.twin.showcase import HOMES, load_showcase  # noqa: E402
from greentwin.twin.simulate import prior_samples  # noqa: E402
from greentwin.twin.twin import GreenDigitalTwin  # noqa: E402

N_SAMPLES = 1000
app = FastAPI(title="GreenTwin dev twin server")
twin = GreenDigitalTwin(n_samples=N_SAMPLES)


class RunRequest(BaseModel):
    property_id: str
    retrofit_ids: list[str] = Field(default_factory=list)
    appliances: list[AppliancePlacement] | None = Field(None, description="None = the showcase home's own list")


class RunResponse(BaseModel):
    baseline_rooms: EnergyOverlay
    scenario: ScenarioView


@app.get("/api/twin/health")
def health() -> dict:
    return {"engine": get_run_physics()[1], "homes": sorted(HOMES), "retrofits": sorted(catalogue())}


@app.post("/api/twin/run", response_model=RunResponse)
def run(req: RunRequest) -> RunResponse:
    if req.property_id not in HOMES:
        raise HTTPException(404, f"unknown home {req.property_id!r}; the dev server knows {sorted(HOMES)}")
    unknown = [r for r in req.retrofit_ids if r not in catalogue()]
    if unknown:
        raise HTTPException(422, f"unknown retrofits {unknown}")
    sc = load_showcase(req.property_id)
    appliances = req.appliances if req.appliances is not None else sc.appliances
    pkg = build_physics_inputs(sc.prop, layout=sc.layout, appliances=appliances, intake=sc.intake, usage=sc.usage)
    samples = prior_samples(pkg.params, N_SAMPLES, priors=pkg.prior_table())  # seeded: same samples every call
    try:
        baseline = run_scenario(twin, sc.layout, samples, [], appliances, weather_source=pkg.weather.source)
        scenario = baseline if not req.retrofit_ids else run_scenario(
            twin, sc.layout, samples, req.retrofit_ids, appliances, weather_source=pkg.weather.source)
    except (KeyError, ValueError) as e:
        raise HTTPException(422, str(e)) from e
    return RunResponse(baseline_rooms=baseline.rooms_after, scenario=scenario)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8765)
