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

from greentwin.twin.geometry import GeometryConfirmation  # noqa: E402
from greentwin.twin.inputs import build_physics_inputs  # noqa: E402
from greentwin.twin.physics_api import get_run_physics  # noqa: E402
from greentwin.twin.retrofits import catalogue  # noqa: E402
from greentwin.twin.rooms import energy_overlay  # noqa: E402
from greentwin.twin.scenarios import run_scenario  # noqa: E402
from greentwin.twin.scene import (  # noqa: E402
    AppliancePlacement,
    EnergyOverlay,
    RoomScene,
    ScenarioView,
    build_scene,
    retrofit_options,
)
from greentwin.twin.showcase import HOMES, example_ids, load_showcase  # noqa: E402
from greentwin.twin.simulate import prior_samples  # noqa: E402
from greentwin.twin.twin import GreenDigitalTwin  # noqa: E402

N_SAMPLES = 1000
app = FastAPI(title="GreenTwin dev twin server")
twin = GreenDigitalTwin(n_samples=N_SAMPLES)


class RunRequest(BaseModel):
    property_id: str
    retrofit_ids: list[str] = Field(default_factory=list)
    appliances: list[AppliancePlacement] | None = Field(None, description="None = the showcase home's own list")
    geometry: GeometryConfirmation | None = Field(None, description="Confirmed/corrected geometry; None = fitted")


class SceneRequest(BaseModel):
    property_id: str
    geometry: GeometryConfirmation
    appliances: list[AppliancePlacement] | None = None


def _context(property_id: str, appliances, geometry):
    try:
        sc = load_showcase(property_id)  # SH1-SH3, or a house-type example EXAMPLE_<archetype>
    except (KeyError, StopIteration) as e:
        raise HTTPException(404, f"unknown home {property_id!r}; known: {sorted(HOMES) + example_ids()}") from e
    apps = appliances if appliances is not None else sc.appliances
    intake = sc.intake.model_copy(update={"geometry": geometry}) if geometry else sc.intake
    pkg = build_physics_inputs(sc.prop, layout=sc.layout, appliances=apps, intake=intake, usage=sc.usage)
    layout = sc.layout
    if geometry:
        from greentwin.twin.geometry import apply_geometry

        layout, _ = apply_geometry(geometry, sc.prop)
    samples = prior_samples(pkg.params, N_SAMPLES, priors=pkg.prior_table())  # seeded: same samples every call
    return sc, layout, apps, pkg, samples


class RunResponse(BaseModel):
    baseline_rooms: EnergyOverlay
    scenario: ScenarioView


@app.get("/api/twin/health")
def health() -> dict:
    return {"engine": get_run_physics()[1], "homes": sorted(HOMES) + example_ids(), "retrofits": sorted(catalogue())}


@app.post("/api/twin/run", response_model=RunResponse)
def run(req: RunRequest) -> RunResponse:
    unknown = [r for r in req.retrofit_ids if r not in catalogue()]
    if unknown:
        raise HTTPException(422, f"unknown retrofits {unknown}")
    sc, layout, appliances, pkg, samples = _context(req.property_id, req.appliances, req.geometry)
    try:
        baseline = run_scenario(twin, layout, samples, [], appliances, weather_source=pkg.weather.source)
        scenario = baseline if not req.retrofit_ids else run_scenario(
            twin, layout, samples, req.retrofit_ids, appliances, weather_source=pkg.weather.source)
    except (KeyError, ValueError) as e:
        raise HTTPException(422, str(e)) from e
    return RunResponse(baseline_rooms=baseline.rooms_after, scenario=scenario)


@app.post("/api/twin/scene", response_model=RoomScene)
def scene(req: SceneRequest) -> RoomScene:
    """Rebuild the room scene from a confirmed/corrected geometry, with energy for the current appliances."""
    sc, layout, appliances, pkg, samples = _context(req.property_id, req.appliances, req.geometry)
    alloc = twin.room_energy(sc.prop, layout, samples, appliances)
    out = build_scene(layout, req.property_id, appliances)
    out.energy = energy_overlay(alloc, calibrated=False, weather_source=pkg.weather.source)
    out.retrofits = retrofit_options(layout, sc.prop)
    out.geometry = pkg.geometry.model_dump(mode="json")
    out.input_quality = pkg.provenance_summary.model_dump(mode="json")
    return out


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8765)
