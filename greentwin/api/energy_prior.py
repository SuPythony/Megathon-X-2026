"""M04 energy prior over HTTP. A router the main app (P4) mounts with ``app.include_router(router)``.

    POST /m04/energy-prior          Property                      -> EnergyPrior (contract)
    POST /m04/energy-prior/update   {property, usage[, imd_quintile]} -> prior + Bayesian posterior per fuel
    GET  /m04/health                model version, data version, fitted Bayes metadata

Run standalone for a quick look:  uv run uvicorn greentwin.api.energy_prior:app --port 8040
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, Field

from contracts.models import EnergyPrior, EnergyUsageMonth, Property
from greentwin.models.energy_prior import MODEL_ID, MODEL_VERSION, load_default
from greentwin.models.prior_update import update_with_meters

router = APIRouter(prefix="/m04", tags=["M04 energy prior"])


class UpdateRequest(BaseModel):
    property: Property
    usage: list[EnergyUsageMonth] = Field(default_factory=list, description="Monthly readings; null = missing")
    imd_quintile: Optional[int] = Field(None, ge=1, le=5)


@router.post("/energy-prior", response_model=EnergyPrior)
def energy_prior(prop: Property, imd_quintile: Optional[int] = None) -> EnergyPrior:
    """Prior for a home without meter data: per-fuel Distribution + additive EBM contributions."""
    return load_default().energy_prior(prop, imd_quintile)


@router.post("/energy-prior/update")
def energy_prior_update(req: UpdateRequest) -> dict:
    """Prior updated with the home's own meter readings (latest 12 months; missing months imputed, not 0)."""
    model = load_default()
    try:
        update = update_with_meters(model, req.property, req.usage)
    except ValueError as e:  # e.g. Bayes layer not fitted
        raise HTTPException(status_code=503, detail=str(e)) from e
    prior = model.energy_prior(req.property, req.imd_quintile)
    return {"prior": prior.model_dump(mode="json"), "update": update}


@router.get("/health")
def health() -> dict:
    m = load_default()
    return {"model": f"{MODEL_ID}-{MODEL_VERSION}", "data_version": m.metadata.get("data_version"),
            "fuels": sorted(m.models), "bayes_layer": all(k in m.metadata for k in ("noise", "area_elasticity")),
            "noise": m.metadata.get("noise")}


app = FastAPI(title="GreenTwin M04 energy prior")
app.include_router(router)
