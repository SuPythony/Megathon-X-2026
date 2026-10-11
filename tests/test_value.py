"""Property value (Part 4): guarantees the judges check (indicative wording, current value, unavailable cases)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from contracts.models import Property, RatingEstimate
from contracts.summarise import exact, summarise
from greentwin.interfaces import ValueModel
from greentwin.value.model import HedonicValueModel, ppd_type
from greentwin.value.prepare import PrepConfig, prepare

EX = Path(__file__).resolve().parents[1] / "contracts" / "examples"
WORDING = "Indicative range based on historical associations in sales data; not a promised or proven increase."


@pytest.fixture
def fake_artifacts(tmp_path):
    """Tiny posterior: beta = 0.02 per 10 SAP points (~0.2% per point) for NW semis, pooled elsewhere."""
    rng = np.random.default_rng(0)
    n = 500
    rt = ["E12000002|semi", "E12000009|semi"]
    np.savez(tmp_path / "posterior.npz", beta_rt=np.column_stack([rng.normal(0.02, 0.002, n)] * 2 + [np.zeros(n)]),
             beta0=rng.normal(0.02, 0.002, n))
    (tmp_path / "encoding.json").write_text(json.dumps({"rt": rt}))
    return str(tmp_path)


def load(pid):
    return Property.model_validate_json((EX / f"{pid}_property.json").read_text(encoding="utf-8"))


def rating(mean, sd=2.0, band="C"):
    return RatingEstimate(sap=summarise(np.random.default_rng(1).normal(mean, sd, 1000), "SAP points"), band=band,
                          anchored_to_recorded_sap=True)


def test_interface(fake_artifacts):
    assert isinstance(HedonicValueModel(artifact_dir=fake_artifacts), ValueModel)


def test_wording_label_and_current_value(fake_artifacts):
    p = load("SH2")  # SAP 60, value 325k
    v = HedonicValueModel(artifact_dir=fake_artifacts).value_impact(p, rating(72))
    assert v.label == "indicative" and v.wording == WORDING
    assert v.current_value_gbp == p.current_value_estimate_gbp  # baseline = current estimate, never a sale price
    # +12 SAP at ~0.2%/point -> ~2.4%
    assert 1.5 < v.uplift_pct.median < 3.5
    assert abs(v.value_after_gbp.median / p.current_value_estimate_gbp - 1 - v.uplift_pct.median / 100) < 0.002


def test_triangulated_mixture_spans_designs(fake_artifacts):
    eff = {"designs": {"low": {"type": "normal", "mean": 0.0008, "sd": 1e-5},
                       "high": {"type": "normal", "mean": 0.0025, "sd": 1e-5},
                       "bayes": {"type": "posterior"}},
           "weights": {"low": 1 / 3, "high": 1 / 3, "bayes": 1 / 3}}
    (Path(fake_artifacts) / "rating_effects.json").write_text(json.dumps(eff))
    v = HedonicValueModel(artifact_dir=fake_artifacts).value_impact(load("SH2"), rating(80, sd=0.01))  # dSAP 20
    lo, hi = v.uplift_pct.interval(0.9).lo, v.uplift_pct.interval(0.9).hi
    assert lo < 1.8 and hi > 4.5  # 0.08%/pt x 20 = 1.6% ... 0.25%/pt x 20 = 5.1%


def test_no_change_no_uplift(fake_artifacts):
    p = load("SH2")
    v = HedonicValueModel(artifact_dir=fake_artifacts).value_impact(
        p, RatingEstimate(sap=exact(60, "SAP points", 10), band="D", anchored_to_recorded_sap=True))
    assert abs(v.uplift_pct.median) < 1e-9


def test_unavailable_without_rating_or_recorded_sap(fake_artifacts):
    m = HedonicValueModel(artifact_dir=fake_artifacts)
    p = load("SH2")
    v = m.value_impact(p, None)
    assert v.method == "unavailable" and v.unavailable_reason and v.uplift_pct is None
    q = p.model_copy(deep=True)
    q.epc.sap_score = None
    v2 = m.value_impact(q, rating(72))
    assert v2.method == "unavailable" and v2.uplift_pct is None


def test_green_twins_ranges_only_min_sales_and_empty_ok():
    from greentwin.value.green_twins import MIN_SALES, find, home_score
    p = load("SH1")
    own = home_score(p)
    rows = []
    for i in range(MIN_SALES):  # exactly MIN_SALES greener comparable sales in the LA
        rows.append({"ptype": "terraced", "year": 2022, "tfa": p.floor_area_m2, "BUILT_FORM": "Mid-Terrace",
                     "age_band": "1900-1929", "lsoa21": "E01", "msoa21": "E02", "laua": "E08000003",
                     "rgn": "E12000002", "green_score": own + 20, "price_adj": 250000 + 1000 * i,
                     "CURRENT_ENERGY_RATING": "C"})
    s = pd.DataFrame(rows)
    twins = find(p, s)
    assert len(twins) == 1 and twins[0].price_per_m2_gbp.n_samples == MIN_SALES
    assert "local authority" in twins[0].same
    assert not any(k in twins[0].model_dump_json() for k in ("paon", "street", "postcode"))
    assert find(p, s.iloc[: MIN_SALES - 1]) == []  # fewer than MIN_SALES -> nothing shown


@pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / "artifacts/value_model/v1/posterior.npz").exists(),
                    reason="needs the fitted v1 posterior")
def test_missing_value_gives_full_predictive_distribution():
    p = load("SH2")
    p.current_value_estimate_gbp = None
    v = HedonicValueModel().value_impact(p, rating(72))
    cv = v.current_value_distribution
    assert cv is not None and cv.std > 0 and cv.interval(0.9).lo < cv.median < cv.interval(0.9).hi
    assert 100_000 < cv.median < 1_000_000                 # plausible for a 95 m2 South West semi
    assert v.value_after_gbp is not None and v.value_after_gbp.std > 0
    assert v.value_after_gbp.median > cv.median * 0.99     # joint draws: after = value x (1 + uplift)


def test_type_mapping():
    assert ppd_type(load("SH1")) == "terraced" and ppd_type(load("SH2")) == "semi" and ppd_type(load("SH3")) == "flat"


def test_prepare_drops_epc_after_sale_and_never_keeps_address():
    sales = pd.DataFrame({
        "postcode": ["M14 5AA", "M14 5AA"], "transactionid": ["a", "b"], "price": ["200000", "210000"],
        "dateoftransfer": ["2020-06-01", "2015-06-01"], "propertytype": ["S", "S"], "oldnew": ["N", "N"],
        "duration": ["F", "F"], "categorytype": ["A", "A"], "recordstatus": ["A", "A"], "lad23cd": ["E08000003"] * 2,
        "CURRENT_ENERGY_EFFICIENCY": ["60", "60"], "LODGEMENT_DATE": ["2019-01-01", "2019-01-01"],
        "TOTAL_FLOOR_AREA": ["90", "90"], "NUMBER_HABITABLE_ROOMS": ["5", "5"], "BUILT_FORM": ["Semi-Detached"] * 2,
        "CONSTRUCTION_AGE_BAND": ["England and Wales: 1930-1949"] * 2, "prop_key": ["k1", "k1"],
    })
    nspl = pd.DataFrame({"pcds": ["M14 5AA"], "laua": ["E08000003"], "rgn": ["E12000002"], "ctry": ["E92000001"],
                         "lsoa21": ["E01005000"], "msoa21": ["E02000900"], "lat": [53.4471], "long": [-2.2222],
                         "imd": [5000], "ru11ind": ["A1"]})
    hpi = pd.DataFrame({"month": ["2020-06", "2015-06", "2026-07"], "RegionName": ["Manchester"] * 3,
                        "AreaCode": ["E08000003"] * 3, "Index": [100, 90, 130], "DetachedIndex": [100, 90, 130],
                        "SemiDetachedIndex": [100, 90, 130], "TerracedIndex": [100, 90, 130],
                        "FlatIndex": [100, 90, 130]})
    out, log = prepare(sales, nspl, hpi, PrepConfig())
    assert len(out) == 1 and out["transactionid"].iloc[0] == "a"   # 2015 sale had a 2019 EPC -> dropped (V-05)
    assert abs(out["price_adj"].iloc[0] - 200000 * 1.3) < 1e-6      # HPI to the reference month (V-07)
    assert "postcode" not in out and out["postcode_district"].iloc[0] == "M14"  # V-06
    assert not {"paon", "saon", "street"} & set(out.columns)
