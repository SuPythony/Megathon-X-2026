"""M04 energy prior: feature mapping, contract output, monotonicity (physics), structural zeros."""
from pathlib import Path

import numpy as np
import pytest

from contracts.models import EnergyPrior, Property
from greentwin.models.energy_prior import DEFAULT_ARTIFACT, EnergyPriorModel, prepare_frame, property_to_features

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / DEFAULT_ARTIFACT


def _prop(sh: str) -> Property:
    return Property.model_validate_json((ROOT / f"contracts/examples/{sh}_property.json").read_text())


def test_feature_mapping_showcase():
    sh1, sh2, sh3 = (property_to_features(_prop(s)).iloc[0] for s in ("SH1", "SH2", "SH3"))
    assert (sh1.prop_type, sh1.age_band, sh1.region, sh1.main_heat_fuel) == ("Mid terrace", 1, "North West", "gas")
    assert (sh2.prop_type, sh2.age_band, sh2.floor_band, sh2.epc_band) == ("Semi detached", 2, 2, 3)
    assert (sh3.prop_type, sh3.main_heat_fuel, sh3.age_band) == ("Flat", "not_gas", 4)
    # Missing stays missing (NaN), never 0.
    assert np.isnan(sh1.imd_quintile) and np.isnan(sh1.conservatory)


needs_model = pytest.mark.skipif(not ARTIFACT.exists(), reason="train first: scripts/train_energy_prior.py")


@pytest.fixture(scope="module")
def model():
    return EnergyPriorModel.load(ARTIFACT)


@needs_model
@pytest.mark.parametrize("sh", ["SH1", "SH2", "SH3"])
def test_prior_contract_and_provenance(model, sh):
    prior = EnergyPrior.model_validate(model.energy_prior(_prop(sh)).model_dump())
    assert prior.provenance.engine == "ebm"
    assert prior.electricity_kwh_m2.unit == "kWh/m2/yr"
    lo, hi = prior.electricity_kwh_m2.interval(0.9).lo, prior.electricity_kwh_m2.interval(0.9).hi
    assert 0 <= lo < prior.electricity_kwh_m2.median < hi


@needs_model
def test_no_gas_supply_is_structural_zero(model):
    prior = model.energy_prior(_prop("SH3"))  # all-electric flat
    assert prior.gas_kwh_m2.mean == 0 and prior.gas_kwh_m2.std == 0 and prior.contributions_gas == []


@needs_model
@pytest.mark.parametrize("fuel", ["electricity", "gas"])
def test_contributions_sum_to_prediction(model, fuel):
    feats = property_to_features(_prop("SH2"))
    total = sum(c.contribution for c in model.contributions(feats, fuel))
    assert total == pytest.approx(float(model.models[fuel].predict(prepare_frame(feats, fuel))[0]), abs=0.05)


@needs_model
@pytest.mark.parametrize("edit, fuel", [
    (lambda p: p.walls.__setattr__("insulation", "cavity_filled"), "gas"),
    (lambda p: p.roof.__setattr__("loft_insulation_mm", 300), "gas"),
    (lambda p: p.epc.__setattr__("current_band", "B"), "gas"),
    (lambda p: p.renewables.__setattr__("pv_kwp", 4.0), "electricity"),
])
def test_better_state_never_raises_prior(model, edit, fuel):
    """Physics: insulation / better EPC never raise gas; PV never raises grid import."""
    for sh in ("SH1", "SH2"):
        before = _prop(sh)
        after = before.model_copy(deep=True)
        edit(after)
        f_before, f_after = property_to_features(before), property_to_features(after)
        assert model.predict_point(f_after, fuel)[0] <= model.predict_point(f_before, fuel)[0] + 1e-9


SPLIT_TABLE = ROOT / "data/processed/need_intensity.parquet"


@pytest.mark.skipif(not SPLIT_TABLE.exists(), reason="build first: python -m greentwin.data.need")
def test_split_is_stratified_and_disjoint():
    import pandas as pd

    df = pd.read_parquet(SPLIT_TABLE, columns=["property_id", "prop_type", "age_band", "floor_band", "region",
                                               "main_heat_fuel", "split"])
    assert df.property_id.is_unique  # one row per property -> splits are disjoint by construction
    share = df.split.value_counts(normalize=True)
    for name, target in {"train": 0.6, "tune": 0.1, "cal": 0.1, "test": 0.2}.items():
        assert share[name] == pytest.approx(target, abs=0.002)
    for col in ["prop_type", "age_band", "floor_band", "region", "main_heat_fuel"]:
        t = pd.crosstab(df[col], df.split, normalize="columns")
        assert (t.max(axis=1) - t.min(axis=1)).max() < 0.002  # every stratum level within 0.2 pp across splits
