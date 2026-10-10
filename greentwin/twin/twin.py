"""``GreenDigitalTwin``: Part 1's implementation of the ``DigitalTwin`` Protocol (``greentwin/interfaces.py``).

Other parts call the twin only through these methods. What each one does today:

- params_from_property  built: EPC + fitted layout + RdSAP/SAP defaults (``params.py``)
- apply_retrofits       built: catalogue edits, with a provisional key translation (``retrofits.py``)
- simulate              built: vmap over samples, real physics if present else stub (``simulate.py``)
- calibrate             prior_only when there are no meter readings; full Bayesian calibration is the calibration
                        team's (their inputs: ``physics_inputs``)
- element_heat_loss     built
- electricity_profile   not built (ownership open)
- scene                 built: contract ``TwinScene`` (the richer room scene is ``scene.build_scene``)
- capture               built: Claude Sonnet 5.5 vision draft of the floor plan (``vision.py``)

Provenance: results from the stub carry engine ``mock`` (model_version ``physics_stub/...``), so the existing
"no mock in the demo" rule keeps them out of the demo automatically.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping, Optional, Sequence

import numpy as np

from contracts.models import (
    CalibrationDiagnostics,
    CalibrationResult,
    CaptureResult,
    ElectricityProfileSummary,
    ElementHeatLoss,
    EnergyPrior,
    EnergyUsageMonth,
    FeasibilityReport,
    LoadProfileArchetype,
    Property,
    Provenance,
    RetrofitAction,
    SceneSurface,
    TwinParams,
    TwinScene,
    WeatherMonth,
)
from contracts.summarise import summarise
from greentwin.interfaces import PosteriorSamples, TwinRun
from greentwin.twin.appliances import summarise_appliances
from greentwin.twin.archetypes import TwinLayout, build_layout
from greentwin.twin.inputs import PhysicsInputs, build_physics_inputs
from greentwin.twin.intake import EPCLookup, NoEPCLookup, TwinIntake
from greentwin.twin.params import ExtendedTwinParams, choose_archetype, fuel_config, params_with_sources
from greentwin.twin.retrofits import apply_retrofits
from greentwin.twin.rooms import RoomAllocation, allocate_rooms
from greentwin.twin.scene import AppliancePlacement
from greentwin.twin.simulate import prior_samples, simulate
from greentwin.twin.weather import resolve_weather

MODEL_VERSION = "twin/0.1"
ELEMENT_FIELD = {"walls": "wall", "roof": "roof", "floor": "floor", "windows": "window", "doors": "door"}


def provenance(engine: str, n_samples: int, seed: int = 42, data_version: str = "synthetic-weather") -> Provenance:
    contract_engine = "physics_l1" if engine == "physics" else "mock"
    return Provenance(engine=contract_engine, model_version=f"{engine}/{MODEL_VERSION}", data_version=data_version,
                      seed=seed, n_samples=n_samples)


def element_heat_loss_from_run(run: TwinRun, params: TwinParams) -> list[ElementHeatLoss]:
    total = sum(np.asarray(v, dtype=float) for v in run.element_heat_loss_w_per_k.values())
    safe = np.where(total > 0, total, 1.0)
    out = []
    for element, values in run.element_heat_loss_w_per_k.items():
        v = np.asarray(values, dtype=float)
        field = ELEMENT_FIELD.get(element)
        out.append(ElementHeatLoss(
            element=element, heat_loss_w_per_k=round(float(v.mean()), 2),
            heat_loss_distribution=summarise(v, "W/K", worst_tail="high") if np.ptp(v) > 0 else None,
            share_pct=round(float(np.mean(v / safe)) * 100, 1),
            u_value=getattr(params, f"{field}_u") if field else None,
            area_m2=getattr(params, f"{field}_area_m2") if field else None,
        ))
    return sorted(out, key=lambda e: -e.heat_loss_w_per_k)


class GreenDigitalTwin:
    """Implements ``greentwin.interfaces.DigitalTwin``."""

    def __init__(self, n_samples: int = 1000, seed: int = 42, epc_lookup: EPCLookup = NoEPCLookup()):
        self.n_samples = n_samples
        self.seed = seed
        self.epc_lookup = epc_lookup
        self.last_engine = "physics_stub"

    # ---------------------------------------------------------------- Protocol methods
    def params_from_property(self, prop: Property, layout: Optional[TwinLayout] = None) -> ExtendedTwinParams:
        return params_with_sources(prop, layout)[0]

    def apply_retrofits(self, params: TwinParams, actions: Sequence[RetrofitAction]) -> TwinParams:
        return apply_retrofits(params, actions)

    def simulate(self, samples: PosteriorSamples, actions: Sequence[RetrofitAction], weather: str = "typical",
                 mode: Literal["calibrated", "standard"] = "calibrated",
                 weather_months: Optional[Sequence[WeatherMonth]] = None) -> TwinRun:
        """Retrofit actions are applied to every sample's base parameters, then the package is re-simulated."""
        if weather != "typical" and weather_months is None:
            raise NotImplementedError(f"weather scenario {weather!r} needs Part 0's load_weather; only 'typical' runs today")
        base = samples.base
        if actions:
            new_base = apply_retrofits(base, actions)
            params = dict(samples.params)
            for key, value in new_base.model_dump().items():
                old = getattr(base, key)
                if isinstance(value, (int, float)) and not isinstance(value, bool) and value != old and key in params:
                    # Edited fields: set to the retrofit value (calibrated multipliers stay per sample)
                    params[key] = np.full(samples.n_samples, float(value), dtype=np.float32)
            samples = PosteriorSamples(property_id=samples.property_id, n_samples=samples.n_samples,
                                       params=params, base=new_base, seed=samples.seed)
        district = getattr(samples.base, "postcode_district", "UNKNOWN")
        arrays, _ = resolve_weather(district, weather_months)
        run, self.last_engine = simulate(samples, arrays, fuel_config(samples.base) if isinstance(samples.base, ExtendedTwinParams)
                                         else _fuel_from_base(samples.base), mode)
        return run

    def calibrate(self, prop: Property, usage: Sequence[EnergyUsageMonth], weather: Sequence[WeatherMonth],
                  prior: Optional[EnergyPrior] = None) -> tuple[CalibrationResult, PosteriorSamples]:
        """No meter readings -> prior_only (built). With readings, Bayesian calibration is the calibration team's:
        give them ``physics_inputs(prop, usage=usage)``."""
        params = self.params_from_property(prop)
        if any(r.gas_kwh is not None or r.electricity_import_kwh is not None for r in usage):
            raise NotImplementedError("Bayesian calibration is built by the calibration team; inputs: "
                                      "GreenDigitalTwin.physics_inputs(prop, usage=usage)")
        samples = prior_samples(params, self.n_samples, self.seed)
        calibrated = ("heat_loss_multiplier", "setpoint_c", "baseload_kwh_per_day", "heating_efficiency")
        result = CalibrationResult(
            property_id=prop.property_id, method="prior_only", n_samples=self.n_samples, months_used=0,
            parameters={k: summarise(samples.params[k], "") for k in calibrated},
            diagnostics=CalibrationDiagnostics(), data_confidence="low",
            posterior_ref=f"memory:prior/{prop.property_id}", prior_source="rdsap_defaults",
            provenance=provenance(self.last_engine, self.n_samples, self.seed),
        )
        return result, samples

    def element_heat_loss(self, samples: PosteriorSamples, actions: Sequence[RetrofitAction] = ()) -> list[ElementHeatLoss]:
        run = self.simulate(samples, actions)
        return element_heat_loss_from_run(run, samples.base)

    def electricity_profile(self, samples: PosteriorSamples, actions: Sequence[RetrofitAction],
                            profiles: Sequence[LoadProfileArchetype], tariffs: Mapping[str, Any]) -> ElectricityProfileSummary:
        raise NotImplementedError("half-hourly electricity layer not built; ownership is an open decision")

    def scene(self, prop: Property, samples: PosteriorSamples, actions: Sequence[RetrofitAction] = (),
              feasibility: Optional[FeasibilityReport] = None, layout: Optional[TwinLayout] = None) -> TwinScene:
        layout = layout or build_layout(choose_archetype(prop), total_floor_area_m2=prop.floor_area_m2,
                                        built_form=prop.built_form)
        elements = {e.element: e for e in self.element_heat_loss(samples, actions)}
        surfaces: list[SceneSurface] = []
        ext_by_side: dict[str, float] = {}
        for w in layout.walls:
            if w.kind == "exterior" and w.side:
                ext_by_side[w.side] = ext_by_side.get(w.side, 0.0) + w.length_m * w.height_m
        total_ext = sum(ext_by_side.values()) or 1.0
        for side, area in ext_by_side.items():
            share = elements["walls"].share_pct * area / total_ext if "walls" in elements else 0.0
            azimuth = next(w.orientation_deg for w in layout.walls if w.side == side and w.orientation_deg is not None)
            surfaces.append(SceneSurface(surface_id=f"walls:{side}", element="walls", orientation_deg=azimuth,
                                         area_m2=round(area, 2), u_value=samples.base.wall_u, heat_loss_share_pct=round(share, 1)))
        for element in ("roof", "floor", "windows", "doors"):
            e = elements.get(element)
            if e and e.area_m2:
                surfaces.append(SceneSurface(surface_id=element, element=element, area_m2=e.area_m2, u_value=e.u_value or 0.0,
                                             heat_loss_share_pct=e.share_pct))
        return TwinScene(property_id=prop.property_id, geometry_source="parametric", property_type=prop.property_type,
                         built_form=prop.built_form, storeys=layout.storeys, footprint_m=layout.footprint_m,
                         roof_type=prop.roof.type, surfaces=surfaces, rooms=layout.rooms)

    def capture(self, property_id: str, floor_plan_image: Optional[bytes], room_photos: Sequence[bytes]) -> CaptureResult:
        """Floor plan only in v1 (room photos are ignored). A draft: nothing reaches the twin until confirmed."""
        if not floor_plan_image:
            return CaptureResult(property_id=property_id, vision_model=None)
        from greentwin.twin.vision import extract_floor_plan, fit_layout

        extraction = extract_floor_plan(floor_plan_image)
        fitted = fit_layout(extraction)
        plan = fitted.layout.to_floor_plan(property_id)
        plan.source = "haiku_vision"  # contract literal for 'vision draft'; the model used is in vision_model
        return CaptureResult(property_id=property_id, floor_plan=plan,
                             draft_property_fields={"archetype_id": extraction.archetype_id,
                                                    "storeys": extraction.storeys, "bedrooms": extraction.bedrooms},
                             needs_confirmation=fitted.needs_confirmation, vision_model="claude-sonnet-5-5")

    # ---------------------------------------------------------------- Part 1 extras
    def physics_inputs(self, prop: Property, layout: Optional[TwinLayout] = None,
                       appliances: Optional[Sequence[AppliancePlacement]] = None, intake: Optional[TwinIntake] = None,
                       usage: Optional[Sequence[EnergyUsageMonth]] = None,
                       weather_months: Optional[Sequence[WeatherMonth]] = None) -> PhysicsInputs:
        """Everything the physics and calibration teams need for this home (see inputs.py)."""
        return build_physics_inputs(prop, layout, appliances, intake, usage, weather_months)

    def room_energy(self, prop: Property, layout: TwinLayout, samples: PosteriorSamples,
                    appliances: Optional[Sequence[AppliancePlacement]] = None) -> RoomAllocation:
        run = self.simulate(samples, ())
        n = getattr(samples.base, "occupants", None) or 2.4
        summary = summarise_appliances(appliances, n) if appliances else None
        return allocate_rooms(layout, run, samples, summary, engine=self.last_engine)


def _fuel_from_base(base: TwinParams):
    from greentwin.twin.physics_api import FuelConfig

    return FuelConfig(heating_fuel=base.heating_fuel, hot_water_fuel=base.heating_fuel)


__all__ = ["GreenDigitalTwin", "element_heat_loss_from_run", "provenance"]
