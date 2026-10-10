"""Green Home Score (Part 3): S = sum_k w_k * s_k with the brief's fixed weights.

Uncertain inputs (metered intensity, an estimated or missing rating, an unknown loft depth) are sampled; the
formula is evaluated per sample to give ``score_distribution`` and ``band_probabilities``. The reported
``score`` uses the posterior-mean inputs (contract), and ``band`` classifies the rounded score (decision D-01).
"""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np

from contracts.models import (
    AreaEPCStats, EnergyBreakdown, Property, RatingEstimate, RoofType, ScoreComponent, ScoreResult,
)
from contracts.summarise import category_probabilities, summarise

from . import mapping as m
from .issues import key_issues
from .sampling import from_distribution, lighting_prior, loft_mm_prior, peer_band_probs, sap_from_band_mix

SCORE_BANDS = ["Very poor", "Poor", "Moderate", "Good", "Highly efficient"]
PV_SELF_CONSUMPTION_DEFAULT = 0.4  # LLD 4.1.1 default share of PV used on site (for the overlap disclosure)


class GreenHomeScorer:
    """Implements ``greentwin.interfaces.GreenHomeScorer``."""

    def __init__(self, n_samples: int = 2000, seed: int = 42, weights: Optional[Mapping[str, float]] = None,
                 intensity_peers: Optional[Mapping[str, float]] = None, learned_weights_dir: Optional[str] = "auto"):
        self.n = n_samples
        self.seed = seed
        self.w = dict(weights or m.weights())
        if abs(sum(self.w.values()) - 1) > 1e-9:
            raise ValueError("weights must sum to 1")
        self.intensity_peers = intensity_peers  # peer median kWh/m2 keyed by property type (fitted on NEED)
        self.learned = self._load_learned(learned_weights_dir)

    @staticmethod
    def _load_learned(d: Optional[str]) -> Optional[dict]:
        """Learned-weights companion (D-05). 'auto' = latest artifacts/score_weights/v*; None = off."""
        import json
        from pathlib import Path
        if d is None:
            return None
        root = Path(__file__).resolve().parents[2] / "artifacts" / "score_weights"
        path = Path(d) if d != "auto" else max(root.glob("v*/weights.json"), default=None, key=lambda p: p.parent.name)
        if path is None or not Path(path).exists():
            return None
        path = Path(path)
        if path.is_dir():
            path = path / "weights.json"
        j = json.loads(path.read_text(encoding="utf-8"))
        try:
            version = path.parent.relative_to(root.parents[1]).as_posix()
        except ValueError:  # weights file outside the repo (tests, ad-hoc runs)
            version = path.parent.as_posix()
        return {"weights": j["weights"], "version": version, "method": j.get("method", "")}

    # ------------------------------------------------------------------ public
    def score(self, prop: Property, energy: EnergyBreakdown, rating: Optional[RatingEstimate] = None,
              peer: Optional[AreaEPCStats] = None) -> ScoreResult:
        rng = np.random.default_rng(self.seed)
        n = self.n
        missing: list[str] = []

        # --- energy efficiency: recorded SAP, estimated rating, or peer-sampled (D-02)
        if rating is not None:
            sap = from_distribution(rating.sap, n, rng)
            rating_source, projected = "estimated", True
        elif prop.epc.sap_score is not None:
            sap = np.full(n, float(prop.epc.sap_score))
            rating_source, projected = "recorded_epc", False
        elif prop.epc.current_band is not None:
            sap = np.full(n, m.band_midpoint(prop.epc.current_band))
            rating_source, projected = "recorded_epc", False
        else:
            probs, src = peer_band_probs(peer.band_counts if peer else None)
            sap = sap_from_band_mix(probs, n, rng)
            rating_source, projected = "missing", False
            missing.append(f"energy rating (sampled from {src} band mix)")

        # --- intensity: annualised METERED kWh/m2 (brief section 2), resampled from the producer's posterior
        intensity = from_distribution(energy.intensity_kwh_m2, n, rng)

        # --- insulation: unknown loft depth is sampled, never 0 (D-04)
        loft = None
        if prop.roof.type == RoofType.pitched_loft and prop.roof.loft_insulation_mm is None:
            loft = loft_mm_prior(prop.construction_age_band, n, rng)
            missing.append("loft insulation depth")

        # --- lighting (part of smart management, D-26): unknown share is sampled, never 0
        if prop.low_energy_lighting_pct is None:
            light = lighting_prior(n, rng)
            missing.append("low-energy lighting share")
        else:
            light = np.full(n, float(prop.low_energy_lighting_pct))

        samples = {
            "energy_efficiency": m.ee_from_sap(sap),
            "energy_intensity": m.intensity_sub(intensity),
            "insulation": np.broadcast_to(m.insulation_sub(prop, loft), (n,)).astype(float),
            "heating_system": np.full(n, m.heating_sub(prop.heating.main_system, prop.heating.system_age_years)),
            "windows_glazing": np.full(n, m.glazing_sub(prop.glazing)),
            "renewable_energy": np.full(n, m.renewables_sub(prop.renewables.pv_kwp, prop.renewables.battery_kwh,
                                                            prop.renewables.solar_thermal)),
            "smart_energy_management": np.broadcast_to(m.smart_sub(prop.controls, light), (n,)).astype(float),
        }
        # point sub-scores at the posterior-mean inputs (contract: ScoreResult.score)
        point = {
            "energy_efficiency": float(m.ee_from_sap(sap.mean())),
            "energy_intensity": float(m.intensity_sub(intensity.mean())),
            "insulation": float(m.insulation_sub(prop, None if loft is None else loft.mean())),
            "smart_energy_management": float(m.smart_sub(prop.controls, light.mean())),
        }
        for k in m.COMPONENTS:
            point.setdefault(k, float(samples[k][0]))

        total = sum(self.w[k] * samples[k] for k in m.COMPONENTS)
        raw = sum(self.w[k] * point[k] for k in m.COMPONENTS)
        score = m.round_half_up(raw)
        band = m.classify(score)  # D-01: classify the rounded score
        rounded = np.floor(total + 0.5).astype(int)
        _, band_probs = category_probabilities([m.classify(int(s)) for s in rounded], SCORE_BANDS)

        inputs = self._inputs(prop, sap, intensity, loft, energy, rating_source)
        inputs["smart_energy_management"]["controls_level"] = m.controls_level(prop.controls)
        inputs["smart_energy_management"]["low_energy_lighting_pct"] = (
            prop.low_energy_lighting_pct if prop.low_energy_lighting_pct is not None
            else f"sampled, mean {light.mean():.0f}")
        components = []
        for k in m.COMPONENTS:
            s = samples[k]
            uncertain = float(np.ptp(s)) > 0
            components.append(ScoreComponent(
                name=k, weight=self.w[k], sub_score=round(point[k], 2),
                contribution=round(self.w[k] * point[k], 3),
                contribution_distribution=summarise(self.w[k] * s, "points") if uncertain else None,
                inputs_used=inputs[k]))

        alternatives = []
        if self.learned:
            from contracts.models import AlternativeScore
            lw = self.learned["weights"]
            l_total = sum(lw[k] * samples[k] for k in m.COMPONENTS)
            l_score = m.round_half_up(sum(lw[k] * point[k] for k in m.COMPONENTS))
            alternatives.append(AlternativeScore(
                weights=lw, score=l_score, band=m.classify(l_score),
                score_distribution=summarise(l_total, "points"),
                weights_version=self.learned["version"], method=self.learned["method"]))

        peer_p50 = self._peer_p50(prop)
        issues = key_issues(prop, point, intensity, peer_p50, missing + list(prop.data_quality.missing_fields))

        return ScoreResult(
            score=score,
            score_distribution=summarise(total, "points", tails=[("Poor or Very poor", 49.5, "below")]),
            band=band,
            band_probabilities=band_probs,
            p_poor_or_very_poor=round(band_probs["Poor"] + band_probs["Very poor"], 4),
            components=components,
            key_issues=issues,
            projected=projected,
            rating_source=rating_source,
            peer_context=self._peer_context(peer, peer_p50),
            mapping_version=f"score_mapping.json v{m.mapping()['version']}",
            alternatives=alternatives,
        )

    # ------------------------------------------------------------------ helpers
    def _peer_p50(self, prop: Property) -> Optional[float]:
        """Median metered kWh/m2 of NEED homes of the same type and floor-area band (per m2, so size-fair)."""
        if self.intensity_peers is not None:
            return self.intensity_peers.get(prop.property_type.value)
        from .archetypes import epc_need_type
        from .sampling import priors
        cells = priors().get("intensity_peer_p50", {}).get("cells", {})
        t = epc_need_type(prop.property_type.value, prop.built_form.value)
        band = 1 + sum(prop.floor_area_m2 > x for x in (50, 100, 150, 200))
        hit = cells.get(f"{t}|{band}")
        return hit["p50"] if hit else None

    def _inputs(self, prop, sap, intensity, loft, energy, rating_source) -> dict[str, dict]:
        pv_note = self._pv_overlap(prop, energy)
        return {
            "energy_efficiency": {"sap_mean": round(float(sap.mean()), 1), "rating_source": rating_source},
            "energy_intensity": {"kwh_m2_mean": round(float(intensity.mean()), 1), **pv_note},
            "insulation": {"walls": f"{prop.walls.type.value}/{prop.walls.insulation.value}",
                           "roof": prop.roof.type.value,
                           "loft_mm": prop.roof.loft_insulation_mm if loft is None else f"sampled, mean {loft.mean():.0f}",
                           "floor_insulated": prop.floor.insulated},
            "heating_system": {"system": prop.heating.main_system.value, "age_years": prop.heating.system_age_years},
            "windows_glazing": {g.type.value: g.share_pct for g in prop.glazing},
            "renewable_energy": {"pv_kwp": prop.renewables.pv_kwp, "battery_kwh": prop.renewables.battery_kwh,
                                 "solar_thermal": str(prop.renewables.solar_thermal)},
            "smart_energy_management": {"controls": ", ".join(c.value for c in prop.controls) or "none"},
        }

    def _pv_overlap(self, prop, energy) -> dict:
        """Disclose the PV overlap: intensity sub-score with self-consumed PV added back (score_mapping.json)."""
        if prop.renewables.pv_kwp <= 0 or energy.pv_generation_kwh is None:
            return {"pv_overlap_points": 0.0}
        self_kwh = energy.pv_generation_kwh.mean * PV_SELF_CONSUMPTION_DEFAULT
        with_pv = energy.intensity_kwh_m2.mean + self_kwh / prop.floor_area_m2
        delta = float(m.intensity_sub(energy.intensity_kwh_m2.mean) - m.intensity_sub(with_pv))
        return {"pv_self_consumption_kwh_m2": round(self_kwh / prop.floor_area_m2, 1),
                "pv_overlap_points": round(self.w["energy_intensity"] * delta, 2)}

    @staticmethod
    def _peer_context(peer: Optional[AreaEPCStats], peer_p50) -> dict:
        ctx: dict = {}
        if peer and peer.band_counts:
            tot = sum(peer.band_counts.get(b, 0) for b in "ABCDEFG")
            if tot:
                ctx["la_share_epc_a_to_c"] = round(sum(peer.band_counts.get(b, 0) for b in "ABC") / tot, 3)
                ctx["source"] = f"EPC live table {'/'.join(peer.source_tables) or 'D1'}, {peer.area_code}, {peer.period}"
        if peer_p50 is not None:
            ctx["peer_intensity_p50_kwh_m2"] = round(float(peer_p50), 1)
        return ctx
