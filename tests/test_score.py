"""Green Home Score: monotonic sub-scores, fixed weights, rounding, missing-data handling, showcase homes."""

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from contracts.models import (
    AreaEPCStats, Control, EnergyUsageMonth, Glazing, GlazingType, HeatingSystem, Property, RatingEstimate,
    WallInsulation, WallType,
)
from contracts.summarise import summarise
from greentwin.interfaces import GreenHomeScorer as ScorerProtocol
from greentwin.score import GreenHomeScorer
from greentwin.score import mapping as m
from greentwin.score.stub_energy import energy_from_usage

EX = Path(__file__).resolve().parents[1] / "contracts" / "examples"


def load(pid):
    prop = Property.model_validate_json((EX / f"{pid}_property.json").read_text(encoding="utf-8"))
    usage = [EnergyUsageMonth.model_validate(r) for r in json.loads((EX / f"{pid}_usage_2025.json").read_text())]
    return prop, usage


def nonincreasing(ys):
    return all(b <= a + 1e-9 for a, b in zip(ys, ys[1:]))


def nondecreasing(ys):
    return all(b >= a - 1e-9 for a, b in zip(ys, ys[1:]))


# ------------------------------------------------------------------ monotonicity sweeps (mandatory)
def test_ee_and_intensity_monotonic():
    assert nondecreasing(m.ee_from_sap(np.arange(-20, 130)))
    assert nonincreasing(m.intensity_sub(np.linspace(0, 800, 2001)))


def test_loft_and_pv_monotonic():
    assert nondecreasing(m.loft_sub(np.arange(0, 500)))
    assert nondecreasing([m.renewables_sub(k) for k in np.linspace(0, 10, 101)])
    for kwp in (0, 2, 6):  # adding a battery or solar thermal never lowers the score
        base = m.renewables_sub(kwp)
        assert m.renewables_sub(kwp, battery_kwh=5) >= base
        assert m.renewables_sub(kwp, solar_thermal=True) >= base


def test_u_value_curves_monotonic_and_consistent_with_loft_table():
    u = np.linspace(0.05, 3.0, 600)
    for f in (m.wall_u_sub, m.roof_u_sub, m.floor_u_sub):
        assert nonincreasing(f(u)), f.__name__
    # RdSAP Table 16: 270 mm <-> U 0.16, 100 mm <-> U 0.40; both paths give the same sub-score
    assert abs(float(m.roof_u_sub(0.16)) - float(m.loft_sub(270))) < 1e-9
    assert abs(float(m.roof_u_sub(0.40)) - float(m.loft_sub(100))) < 1e-9


def test_loft_curve_is_share_of_heat_loss_cut():
    """v0.3.0 (D-24): sub-score = 100 x (U0 - U(mm)) / (U0 - Umin), U from RdSAP 10 Table 16."""
    table = {0: 2.3, 50: 0.68, 100: 0.40, 200: 0.21, 270: 0.16, 400: 0.11}
    for mm, u in table.items():
        assert abs(float(m.loft_sub(mm)) - 100 * (2.3 - u) / (2.3 - 0.11)) < 0.01, mm
    assert float(m.loft_sub(100)) > 85  # first 100 mm delivers most of the benefit


def test_good_loft_does_not_hide_unfilled_cavity():
    prop, usage = load("SH2")  # 100 mm loft, unfilled cavity walls
    res = GreenHomeScorer(n_samples=300).score(prop, energy_from_usage(prop, usage, n=300))
    issue = next(i for i in res.key_issues if i.code == "poor_insulation")
    assert "walls" in issue.label and issue.evidence["walls_sub_score"] == 25


def test_wall_insulation_never_lowers_score():
    better = [WallInsulation.none, WallInsulation.cavity_filled, WallInsulation.internal, WallInsulation.external]
    for wt in WallType:
        for age in ("1900-1929", "1983-1990", "2012 onwards"):
            vals = [m.wall_sub(wt, ins, age) for ins in better]
            assert nondecreasing(vals), (wt, age, vals)
            assert m.wall_sub(wt, WallInsulation.as_built, age) >= m.wall_sub(wt, WallInsulation.none, age)


def test_glazing_upgrade_never_lowers_score():
    order = [GlazingType.single, GlazingType.secondary, GlazingType.double_pre_2002,
             GlazingType.double_post_2002, GlazingType.triple]
    assert nondecreasing([m.glazing_sub([Glazing(type=t)]) for t in order])
    # replacing any share of a worse type by a better type never lowers it
    for share in range(0, 101, 10):
        mixed = [Glazing(type=GlazingType.single, share_pct=100 - share), Glazing(type=GlazingType.triple, share_pct=share)]
        if share:
            assert m.glazing_sub(mixed) >= m.glazing_sub([Glazing(type=GlazingType.single)])


def test_adding_controls_never_lowers_score():
    allc = [c for c in Control if c != Control.none]
    for light in (0, 50, 100):
        for r in range(len(allc) + 1):
            for subset in itertools.combinations(allc, r):
                base = m.smart_sub(subset, light)
                for extra in allc:
                    assert m.smart_sub(set(subset) | {extra}, light) >= base


def test_more_low_energy_lighting_never_lowers_score():
    for ctl in ([], [Control.programmer], [Control.smart_thermostat, Control.zoned_control, Control.smart_meter]):
        assert nondecreasing(m.smart_sub(ctl, np.arange(0, 101)))


def test_missing_lighting_is_sampled_and_controls_issue_ignores_lighting():
    prop, usage = load("SH2")  # lighting not recorded
    assert prop.low_energy_lighting_pct is None
    res = GreenHomeScorer(n_samples=500).score(prop, energy_from_usage(prop, usage, n=300))
    smart = next(c for c in res.components if c.name == "smart_energy_management")
    assert smart.contribution_distribution is not None  # sampled -> uncertain
    assert "sampled" in str(smart.inputs_used["low_energy_lighting_pct"])
    full = prop.model_copy(deep=True)
    full.low_energy_lighting_pct = 100
    res2 = GreenHomeScorer(n_samples=500).score(full, energy_from_usage(full, usage, n=300))
    assert next(c for c in res2.components if c.name == "smart_energy_management").sub_score >= smart.sub_score
    assert not any(i.code == "no_controls" for i in res2.key_issues)  # programmer + room stat = 40 -> not flagged


def test_newer_boiler_never_scores_lower():
    for hs in HeatingSystem:
        ys = [m.heating_sub(hs, a) for a in range(0, 60)]
        assert nonincreasing(ys), hs


def test_every_heating_system_mapped():
    for hs in HeatingSystem:
        assert 0 <= m.heating_sub(hs, None) <= 100


# ------------------------------------------------------------------ formula rules
def test_rounding_is_half_up():
    assert m.round_half_up(48.5) == 49 and m.round_half_up(49.49) == 49 and round(48.5) == 48


def test_weights_fixed_and_contributions_sum_to_score():
    prop, usage = load("SH2")
    res = GreenHomeScorer(n_samples=500).score(prop, energy_from_usage(prop, usage, n=500))
    assert {c.name: c.weight for c in res.components} == m.weights()
    raw = sum(c.contribution for c in res.components)
    assert abs(raw - res.score) <= 0.5 + 1e-6
    assert res.band == m.classify(res.score)          # D-01
    assert abs(sum(res.band_probabilities.values()) - 1) < 1e-9


def test_interface_conformance():
    assert isinstance(GreenHomeScorer(), ScorerProtocol)


# ------------------------------------------------------------------ missing data never becomes 0
def test_missing_rating_sampled_from_peers_and_widens_interval():
    prop, usage = load("SH2")
    energy = energy_from_usage(prop, usage, n=500)
    known = GreenHomeScorer(n_samples=2000).score(prop, energy)
    gone = prop.model_copy(deep=True)
    gone.epc.sap_score, gone.epc.current_band = None, None
    peer = AreaEPCStats(area_code="E06000025", area_type="local_authority", period="2025", lodgements=100,
                        band_counts={"B": 10, "C": 40, "D": 40, "E": 10}, source_tables=["D1"], release="test")
    res = GreenHomeScorer(n_samples=2000).score(gone, energy, peer=peer)
    assert res.rating_source == "missing"
    assert any(i.code == "missing_data" for i in res.key_issues)
    assert res.score_distribution.std > known.score_distribution.std


def test_unknown_loft_is_sampled_not_zero():
    prop, usage = load("SH2")
    energy = energy_from_usage(prop, usage, n=500)
    zero = prop.model_copy(deep=True)
    zero.roof.loft_insulation_mm = 0
    unk = prop.model_copy(deep=True)
    unk.roof.loft_insulation_mm = None
    s0 = GreenHomeScorer(n_samples=1000).score(zero, energy)
    su = GreenHomeScorer(n_samples=1000).score(unk, energy)
    ins = lambda r: next(c for c in r.components if c.name == "insulation")  # noqa: E731
    assert ins(su).sub_score > ins(s0).sub_score
    assert ins(su).contribution_distribution is not None


def test_projected_score_not_lower_for_improvement():
    prop, usage = load("SH1")
    energy = energy_from_usage(prop, usage, n=500)
    before = GreenHomeScorer(n_samples=1000).score(prop, energy)
    rating = RatingEstimate(sap=summarise(np.random.default_rng(0).normal(60, 3, 500), "SAP points"), band="D",
                            anchored_to_recorded_sap=True)
    after = GreenHomeScorer(n_samples=1000).score(prop, energy, rating=rating)
    assert after.projected and after.rating_source == "estimated"
    assert after.score >= before.score


def test_learned_alternative_never_changes_headline(tmp_path):
    prop, usage = load("SH2")
    energy = energy_from_usage(prop, usage, n=500)
    lw = {"energy_efficiency": 0.3, "energy_intensity": 0.1, "insulation": 0.15, "heating_system": 0.2,
          "windows_glazing": 0.05, "renewable_energy": 0.05, "smart_energy_management": 0.15}
    (tmp_path / "weights.json").write_text(json.dumps({"weights": lw, "method": "test"}))
    plain = GreenHomeScorer(n_samples=500, learned_weights_dir=None).score(prop, energy)
    with_alt = GreenHomeScorer(n_samples=500, learned_weights_dir=str(tmp_path)).score(prop, energy)
    assert plain.alternatives == []
    assert with_alt.score == plain.score and with_alt.band == plain.band
    alt = with_alt.alternatives[0]
    expected = sum(lw[c.name] * c.sub_score for c in with_alt.components)
    assert abs(alt.score - expected) <= 0.5 + 0.01 and alt.band == m.classify(alt.score)


@pytest.mark.parametrize("pid", ["SH1", "SH2", "SH3"])
def test_showcase_homes_score(pid):
    prop, usage = load(pid)
    res = GreenHomeScorer(n_samples=1000).score(prop, energy_from_usage(prop, usage, n=500))
    assert 0 <= res.score <= 100 and len(res.components) == 7
