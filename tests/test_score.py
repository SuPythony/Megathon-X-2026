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
    for r in range(len(allc) + 1):
        for subset in itertools.combinations(allc, r):
            base = m.smart_sub(subset)
            for extra in allc:
                assert m.smart_sub(set(subset) | {extra}) >= base


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


@pytest.mark.parametrize("pid", ["SH1", "SH2", "SH3"])
def test_showcase_homes_score(pid):
    prop, usage = load(pid)
    res = GreenHomeScorer(n_samples=1000).score(prop, energy_from_usage(prop, usage, n=500))
    assert 0 <= res.score <= 100 and len(res.components) == 7
