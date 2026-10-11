"""NumPyro home calibration using the existing deterministic JAX physics seam.

Continuous constants are inferred jointly. Unselected synthetic quantities are
marginalized over a frozen joint prior bank, not replaced by their mean. One bank
index describes the entire household/month sequence. Posterior prediction samples
that index conditional on each constant draw and reuses the full physics engine.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import logsumexp
from numpyro.diagnostics import summary as mcmc_summary
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, init_to_median
from numpyro.infer.autoguide import AutoMultivariateNormal
from numpyro.optim import Adam

from contracts.calibration import CalibrationConfig, HomeCalibrationResult, MeterObservation
from contracts.models import Provenance
from contracts.uncertain_twin import GeneratedInputs, PriorMarginal
from greentwin.twin.missing_inputs import (
    InputSamples,
    generate_missing_inputs,
    input_unit,
    moments,
    prepare_electricity_sample,
    sample_marginal,
)
from greentwin.twin.physics import VERSION, run_physics
from greentwin.twin.physics_api import FuelConfig
from greentwin.twin.priors import SYSTEMS, build_priors, request_hash

ENERGY_KEYS = (
    "electricity_import_kwh", "gas_kwh", "oil_kwh", "lpg_kwh", "solid_kwh", "district_kwh",
    "export_kwh", "pv_generation_kwh",
)
MONTHLY_KEYS = {k: "monthly_" + k for k in ENERGY_KEYS}
MONTHLY_KEYS["electricity_import_kwh"] = "monthly_electricity_kwh"
DEFAULT_PARAMETERS = (
    "wall_u", "roof_u", "floor_u", "window_u", "door_u", "air_changes_per_hour",
    "thermal_bridging_y", "heating_efficiency", "hot_water_efficiency",
    "baseload_kwh_per_day", "hot_water_kwh_per_day", "setpoint_c",
)
DEVICE_CONSTANTS = (
    "battery_usable_fraction", "battery_charge_efficiency", "battery_discharge_efficiency",
    "battery_power_c_rate", "inverter_capacity_ratio",
)
INDEPENDENT_HOT_WATER = {"electric_immersion", "instantaneous_electric", "heat_pump_cylinder"}
SUPPORTED = set(DEFAULT_PARAMETERS) | set(DEVICE_CONSTANTS) | {
    "heat_loss_multiplier", "solar_gain_factor", "retained_electric_heat_fraction",
    "heating_auxiliary_power_kw", "cylinder_loss_multiplier", "water_temperature_rise_c",
}


def observations_from_package(package):
    """Read Satyarth's nullable monthly observations without turning gaps into 0."""
    meter = (package or {}).get("meter")
    if not meter:
        return []
    months = meter["months"]
    for key in ("electricity_kwh", "gas_kwh", "estimated"):
        if len(meter[key]) != len(months):
            raise ValueError("Handoff meter columns are not aligned")
    return [
        MeterObservation(month=month, electricity_import_kwh=meter["electricity_kwh"][i],
                         gas_kwh=meter["gas_kwh"][i], estimated=meter["estimated"][i],
                         source=meter.get("source", "Satyarth meter handoff"))
        for i, month in enumerate(months)
    ]


def observations_from_usage(usage, property_id):
    """Convert the existing EnergyUsageMonth contract to home meter observations."""
    rows = []
    for row in usage:
        if row.property_id != property_id:
            raise ValueError("Meter usage belongs to a different household")
        rows.append(MeterObservation(
            month=row.month, electricity_import_kwh=row.electricity_import_kwh,
            gas_kwh=row.gas_kwh, export_kwh=row.electricity_export_kwh,
            pv_generation_kwh=row.pv_generation_kwh, estimated=row.estimated_read,
            source="EnergyUsageMonth contract",
        ))
    return rows


def prior_distribution(spec):
    """Same bounded families/centres as the input generator, in differentiable JAX."""
    if spec.family == "normal":
        return dist.TruncatedNormal(spec.centre, spec.scale, low=spec.lower, high=spec.upper)
    if spec.family == "lognormal":
        mu = np.log(spec.centre) - spec.scale**2 / 2
        low = None if spec.lower is None or spec.lower <= 0 else np.log(spec.lower)
        high = None if spec.upper is None else np.log(spec.upper)
        return dist.TransformedDistribution(
            dist.TruncatedNormal(mu, spec.scale, low=low, high=high), dist.transforms.ExpTransform()
        )
    if spec.family == "beta":
        mean = (spec.centre - spec.lower) / (spec.upper - spec.lower)
        return dist.TransformedDistribution(
            dist.Beta(mean * spec.scale, (1 - mean) * spec.scale),
            dist.transforms.AffineTransform(spec.lower, spec.upper - spec.lower),
        )
    if spec.family == "uniform":
        return dist.Uniform(spec.lower, spec.upper)
    raise ValueError(f"Cannot infer {spec.family} with a continuous sampler")


def _parameter_specs(bank, config):
    requested = config.parameters or list(DEFAULT_PARAMETERS)
    specs, fixed = {}, []
    for key in requested:
        if key not in SUPPORTED:
            raise ValueError(f"Unsupported calibration parameter: {key}")
        if key in bank.prior.facts or (key == "setpoint_c" and "thermostat_c" in bank.prior.facts):
            fixed.append(key)
            continue
        if key in bank.prior.marginals:
            spec = bank.prior.marginals[key]
        elif key in ("baseload_kwh_per_day", "hot_water_kwh_per_day"):
            values = bank.parameters[key]
            if np.std(values) < 1e-10:
                fixed.append(key)
                continue
            spec = PriorMarginal(
                family="normal", centre=float(np.mean(values)), scale=float(np.std(values, ddof=1)),
                lower=0.01, upper=100, unit="kWh/day", role="missing_input",
                source="Moment-matched effective demand prior from the joint synthetic input bank",
                rationale="Infer unresolved effective daily demand; preserve its internal-heat dependency",
            )
        else:
            raise ValueError(f"No initial prior for {key}")
        if spec.family == "exact":
            fixed.append(key)
        else:
            specs[key] = spec
    if not specs:
        raise ValueError("No uncertain selected parameters: exact customer/recorded facts cannot be calibrated")
    if "water_temperature_rise_c" in specs and "hot_water_kwh_per_day" in specs:
        raise ValueError("Choose water temperature rise or effective hot-water demand, not both")
    return specs, fixed


def _apply_constants(p, w, constants, theta, facts):
    """Replace uncertain values and update their physical dependencies together."""
    changed = dict(p)
    weather = dict(w)
    for key, value in theta.items():
        if key in changed:
            changed[key] = value
    if "internal_gains_w" not in facts:
        old_heat = p["retained_electric_heat_fraction"] * p["baseload_kwh_per_day"]
        new_heat = changed["retained_electric_heat_fraction"] * changed["baseload_kwh_per_day"]
        changed["internal_gains_w"] = p["internal_gains_w"] + (new_heat - old_heat) * 1000 / 24
    if "water_temperature_rise_c" in theta:
        ratio = theta["water_temperature_rise_c"] / constants["water_temperature_rise_c"]
        if "hot_water_kwh_per_day" not in facts:
            changed["hot_water_kwh_per_day"] = p["hot_water_kwh_per_day"] * ratio
        if "shower_electric_kwh_per_day" not in facts:
            changed["shower_electric_kwh_per_day"] = p["shower_electric_kwh_per_day"] * ratio
    if "cylinder_loss_multiplier" in theta and "hot_water_standing_loss_kwh_per_day" not in facts:
        changed["hot_water_standing_loss_kwh_per_day"] = (
            p["hot_water_standing_loss_kwh_per_day"]
            * theta["cylinder_loss_multiplier"] / constants["cylinder_loss_multiplier"]
        )
    if "heating_auxiliary_power_kw" in theta and "auxiliary_electricity_kwh_per_day" not in facts:
        weather["heating_auxiliary_electricity_kwh_monthly"] = (
            w["heating_auxiliary_electricity_kwh_monthly"]
            * theta["heating_auxiliary_power_kw"] / constants["heating_auxiliary_power_kw"]
        )
    if "electricity_inputs" in w:
        x = dict(w["electricity_inputs"])
        for key in DEVICE_CONSTANTS[:3]:
            if key in theta:
                x[key] = theta[key]
        if "battery_power_c_rate" in theta:
            ratio = theta["battery_power_c_rate"] / constants["battery_power_c_rate"]
            for key in ("battery_charge_kw", "battery_discharge_kw"):
                if key not in facts:
                    x[key] = x[key] * ratio
        if "inverter_capacity_ratio" in theta and "pv_inverter_kw" not in facts:
            x["pv_inverter_kw"] *= theta["inverter_capacity_ratio"] / constants["inverter_capacity_ratio"]
        weather["electricity_inputs"] = x
    return changed, weather


class CalibrationModel:
    def __init__(self, bank, observations, config, specs):
        self.bank, self.config, self.specs = bank, config, specs
        self.observations = [MeterObservation.model_validate(o) for o in observations]
        if len({o.month for o in self.observations}) != len(self.observations):
            raise ValueError("Duplicate meter months")
        entries = []
        for row in self.observations:
            if row.month not in bank.request.months:
                raise ValueError(f"Meter month {row.month} has no aligned household/weather input")
            if not row.use_for_calibration:
                continue
            for key in ENERGY_KEYS:
                value = getattr(row, key)
                if value is not None:
                    entries.append((bank.request.months.index(row.month), key, value, row.estimated))
        if not entries:
            raise ValueError("At least one nonmissing training meter observation is required")
        self.entries = entries
        self.values = jnp.asarray([entry[2] for entry in entries])
        self.relative_noise = jnp.asarray([
            config.relative_noise_estimated if entry[3] else config.relative_noise_actual for entry in entries
        ])
        self.p = {k: jnp.asarray(a) for k, a in bank.parameters.items()}
        self.w = {k: jnp.asarray(a) for k, a in bank.weather.items()}
        self.constants = {k: jnp.asarray(a) for k, a in bank.constants.items()}
        # With known no-PV/no-battery, interval allocation conserves monthly demand
        # exactly. Use that identical monthly path inside inference, then run full
        # interval dispatch for final outputs. Other cases use the complete solver.
        self.interval_likelihood = bool(
            np.any(bank.parameters["pv_kwp"] > 0) or np.any(bank.parameters["battery_kwh"] > 0)
        )
        if self.interval_likelihood:
            arrays = [prepare_electricity_sample(bank, i).numeric_arrays() for i in range(config.nuisance_samples)]
            self.w["electricity_inputs"] = {
                key: jnp.asarray(np.stack([a[key] for a in arrays])) for key in arrays[0]
            }
        self.fuel = FuelConfig(bank.prior.equipment["heating_fuel"], bank.prior.equipment["hot_water_fuel"])
        self.log_likelihood = jax.jit(self._log_likelihood)

    def predict(self, theta):
        def one(p, w, constants):
            p, w = _apply_constants(p, w, constants, theta, self.bank.prior.facts)
            return run_physics(p, w, "calibrated", self.fuel)
        return jax.vmap(one)(self.p, self.w, self.constants)

    def _log_likelihood(self, theta):
        outputs = self.predict(theta)
        mu = jnp.stack([outputs[MONTHLY_KEYS[key]][:, month] for month, key, _, _ in self.entries], axis=1)
        scale = jnp.sqrt(self.config.absolute_noise_kwh**2 + (self.relative_noise * mu)**2)
        scores = dist.StudentT(self.config.student_df, mu, scale).log_prob(self.values).sum(axis=1)
        return jnp.where(jnp.isfinite(scores), scores, -jnp.inf)

    def __call__(self):
        theta = {key: numpyro.sample(key, prior_distribution(spec)) for key, spec in self.specs.items()}
        scores = self.log_likelihood(theta)
        # Sum each household trajectory's log likelihood before mixing the bank.
        # Mixing each month separately would invent a different home every month.
        numpyro.factor("monthly_meter_likelihood", logsumexp(scores) - jnp.log(len(scores)))


@dataclass
class CalibrationFit:
    result: HomeCalibrationResult
    bank: InputSamples
    samples: dict[str, np.ndarray]  # flat, aligned joint posterior constant draws
    model: CalibrationModel


def scenario_request(fit, changes):
    """Build a coherent proposed state without carrying old device facts forward."""
    facts = copy.deepcopy(fit.bank.request.user_inputs)
    equipment = fit.bank.prior.equipment
    heating_changed = "heating_system" in changes and (
        SYSTEMS.get(changes["heating_system"]) != SYSTEMS.get(equipment["heating_system"])
    )
    water_changed = "hot_water_source" in changes and changes["hot_water_source"] != equipment["hot_water_source"]
    if heating_changed:
        for key in ("heating_efficiency", "heating_auxiliary_power_kw", "auxiliary_electricity_kwh_per_day"):
            if key not in changes:
                facts.pop(key, None)
    if (water_changed or (heating_changed and equipment["hot_water_source"] not in INDEPENDENT_HOT_WATER)) \
            and "hot_water_efficiency" not in changes:
        facts.pop("hot_water_efficiency", None)
    for capacity, flag in (("pv_kwp", "has_solar_pv"), ("battery_kwh", "has_battery")):
        if capacity in changes and flag in facts and flag not in changes:
            facts[flag] = changes[capacity] > 0
    facts.update(changes)
    return type(fit.bank.request).model_validate(fit.bank.request.model_copy(update={"user_inputs": facts}).model_dump())


def calibrate_household(request, observations=None, *, config=None, prior=None):
    """Infer a household posterior; no history writes or modification of user facts."""
    config = config or CalibrationConfig()
    observations = observations if observations is not None else observations_from_package(request.physics_package)
    started = time.perf_counter()
    bank = generate_missing_inputs(request, config.nuisance_samples, prior=prior)
    specs, fixed = _parameter_specs(bank, config)
    model = CalibrationModel(bank, observations, config, specs)
    key = jax.random.PRNGKey(config.seed)
    warnings = [
        "Nuisance integration uses a fixed finite joint prior bank; compare bank sizes for sensitive applications",
        "Monthly bills identify combinations of fabric, ventilation and efficiency; individual constants can remain broad",
    ]
    if config.method == "nuts":
        sampler = MCMC(
            NUTS(model, target_accept_prob=config.target_accept, max_tree_depth=config.max_tree_depth,
                 init_strategy=init_to_median(num_samples=20)),
            num_warmup=config.warmup, num_samples=config.samples, num_chains=config.chains,
            chain_method="sequential", progress_bar=False,
        )
        sampler.run(key, extra_fields=("diverging", "accept_prob", "num_steps"))
        grouped = {k: np.asarray(v) for k, v in sampler.get_samples(group_by_chain=True).items()}
        diagnostic = mcmc_summary(grouped, group_by_chain=True)
        per_parameter = {
            k: {"r_hat": float(v["r_hat"]) if np.isfinite(v["r_hat"]) else None,
                "effective_samples": float(v["n_eff"]) if np.isfinite(v["n_eff"]) else None}
            for k, v in diagnostic.items()
        }
        extra = sampler.get_extra_fields()
        divergences = int(np.asarray(extra["diverging"]).sum())
        diagnostics = {
            "parameters": per_parameter, "divergences": divergences,
            "mean_acceptance": float(np.asarray(extra["accept_prob"]).mean()),
            "tree_depth_limit_hits": int(np.sum(np.asarray(extra["num_steps"]) >= 2**config.max_tree_depth - 1)),
            "chains": config.chains, "draws_per_chain": config.samples,
        }
        converged = config.chains >= 2 and divergences == 0 and diagnostics["tree_depth_limit_hits"] == 0 and all(
            d["r_hat"] is not None and d["r_hat"] <= config.max_r_hat
            and d["effective_samples"] is not None and d["effective_samples"] >= config.min_effective_samples
            for d in per_parameter.values()
        )
        status = "converged" if converged else "needs_review"
        samples = {k: v.reshape(-1) for k, v in grouped.items()}
        if not converged:
            warnings.append("Posterior sampler diagnostics require review before using this fit for decisions")
    else:
        guide = AutoMultivariateNormal(model, init_loc_fn=init_to_median(num_samples=20))
        svi = SVI(model, guide, Adam(config.learning_rate), Trace_ELBO())
        run = svi.run(key, config.svi_steps, progress_bar=False)
        losses = np.asarray(run.losses)
        if not np.isfinite(losses).all():
            raise ValueError("SVI produced nonfinite losses")
        samples = {
            k: np.asarray(v) for k, v in guide.sample_posterior(
                jax.random.fold_in(key, 1), run.params, sample_shape=(config.samples * config.chains,)
            ).items() if k in specs
        }
        diagnostics = {"initial_loss": float(losses[0]), "final_loss": float(losses[-1]),
                       "final_100_mean_loss": float(losses[-100:].mean()), "steps": config.svi_steps,
                       "guide": "full-covariance normal in unconstrained coordinates"}
        status = "approximate"
        warnings.append("SVI is a variational approximation; NUTS chain diagnostics do not apply")
    if any(not np.isfinite(a).all() for a in samples.values()):
        raise ValueError("Nonfinite posterior draws")
    prior_parameters = {
        k: moments(sample_marginal(s, np.random.default_rng(config.seed).uniform(size=4096)), s.unit)
        for k, s in specs.items()
    }
    posterior_parameters = {k: moments(a, specs[k].unit) for k, a in samples.items()}
    if config.method == "nuts":
        for k, d in posterior_parameters.items():
            ess = diagnostics["parameters"][k]["effective_samples"]
            if ess is not None and ess > 0:
                d.monte_carlo_standard_error = d.std / np.sqrt(ess)
    identity = hashlib.sha256()
    identity.update(json.dumps({"request_hash": bank.prior.request_hash, "config": config.model_dump(),
                                "observations": [o.model_dump() for o in model.observations]}, sort_keys=True).encode())
    for k in sorted(samples):
        identity.update(k.encode()); identity.update(samples[k].tobytes())
    for group in (bank.parameters, bank.constants, bank.weather):
        for k in sorted(group):
            identity.update(k.encode()); identity.update(group[k].tobytes())
    result = HomeCalibrationResult(
        property_id=request.property_id, calibration_id=identity.hexdigest(), request_hash=bank.prior.request_hash,
        method=config.method, status=status, config=config, fitted_parameters=list(specs), fixed_parameters=fixed,
        prior_specs=specs, prior_parameters=prior_parameters, posterior_parameters=posterior_parameters,
        observations=model.observations, observation_count=len(model.entries),
        training_months=sorted({o.month for o in model.observations if o.use_for_calibration
                                and any(getattr(o, k) is not None for k in ENERGY_KEYS)}),
        held_out_months=sorted({o.month for o in model.observations if not o.use_for_calibration}),
        diagnostics=diagnostics, elapsed_seconds=time.perf_counter() - started, warnings=warnings,
        provenance=Provenance(engine="physics_l1", model_version=VERSION + "+numpyro-home/0.1.0",
                             data_version=request.data_version, seed=config.seed,
                             n_samples=len(next(iter(samples.values())))),
    )
    return CalibrationFit(result, bank, samples, model)


def predictive_inputs(fit, *, request=None, n_samples=128, posterior=True, allow_needs_review=False):
    """Draw constants and conditional nuisance realizations, then prepare the full twin.

    Repeated calls with the same fit/sample count select the same posterior rows
    and baseline-conditioned nuisance indices. Scenario edits supersede fitted
    constants only for changed components; hypothetical meters are never fitted.
    """
    if not isinstance(n_samples, int) or not 2 <= n_samples <= 10000:
        raise ValueError("Predictive n_samples must be in 2..10000")
    if posterior and fit.result.status == "needs_review" and not allow_needs_review:
        raise ValueError("Calibration diagnostics need review; inspect the fit before using its posterior")
    request = request or fit.bank.request
    if request.property_id != fit.result.property_id:
        raise ValueError("A household posterior cannot be transferred to a different household")
    if request.seed != fit.bank.request.seed:
        raise ValueError("Retain the baseline seed to preserve matched scenario draws")
    rng = np.random.default_rng(fit.result.config.seed + 1009)
    if posterior:
        count = len(next(iter(fit.samples.values())))
        indices = rng.choice(count, size=n_samples, replace=n_samples > count)
        theta = {k: a[indices] for k, a in fit.samples.items()}
        # Restore p(nuisance | theta, baseline meters), not its unconditional prior.
        bank_indices = []
        for i in range(n_samples):
            scores = np.asarray(fit.model.log_likelihood({k: a[i] for k, a in theta.items()}), dtype=float)
            if not np.isfinite(scores).any():
                raise ValueError("No finite conditional nuisance realization for a posterior draw")
            probs = np.exp(scores - np.max(scores)); probs /= probs.sum()
            bank_indices.append(rng.choice(len(probs), p=probs))
        bank_indices = np.asarray(bank_indices)
    else:
        theta = {k: sample_marginal(s, rng.uniform(size=n_samples)) for k, s in fit.result.prior_specs.items()}
        bank_indices = rng.integers(0, fit.result.config.nuisance_samples, size=n_samples)
    same = request_hash(request) == fit.bank.prior.request_hash
    bank = fit.bank
    if not same:
        prior = build_priors(request)
        heating_changed = prior.equipment["heating_system"] != fit.bank.prior.equipment["heating_system"]
        water_changed = (
            (prior.equipment["hot_water_fuel"], prior.equipment["hot_water_source"])
            != (fit.bank.prior.equipment["hot_water_fuel"], fit.bank.prior.equipment["hot_water_source"])
        ) or (heating_changed and prior.equipment["hot_water_source"] not in INDEPENDENT_HOT_WATER)
        # Keep the population prior snapshot for unchanged uncertain components.
        for k, s in fit.bank.prior.marginals.items():
            if (k in prior.marginals and prior.marginals[k].family != "exact"
                    and k not in request.user_inputs
                    and not (k == "heating_efficiency" and heating_changed)
                    and not (k == "hot_water_efficiency" and water_changed)):
                prior.marginals[k] = s.model_copy(deep=True)
        bank = generate_missing_inputs(request, fit.result.config.nuisance_samples, prior=prior)
    active = {k: a for k, a in theta.items() if k not in bank.prior.facts
              and not (k == "setpoint_c" and "thermostat_c" in bank.prior.facts)}
    if not same:
        if heating_changed:
            active.pop("heating_efficiency", None)
        if water_changed:
            active.pop("hot_water_efficiency", None)
    groups = {
        name: {k: a[bank_indices].copy() for k, a in getattr(bank, name).items()}
        for name in ("parameters", "constants", "weather", "profiles", "latent")
    }
    # Replay exactly the bank realization's half-hour profile randomness.
    groups["latent"]["interval_seed_index"] = bank_indices.copy()
    p, w, constants = groups["parameters"], groups["weather"], groups["constants"]
    for i in range(n_samples):
        values = {k: a[i] for k, a in active.items()}
        pp, ww = _apply_constants({k: a[i] for k, a in p.items()}, {k: a[i] for k, a in w.items()},
                                 {k: a[i] for k, a in constants.items()}, values, bank.prior.facts)
        for k in p:
            p[k][i] = pp[k]
        for k in w:
            w[k][i] = ww[k]
        for k, value in values.items():
            if k in constants:
                constants[k][i] = value
            if k in groups["latent"]:
                groups["latent"][k][i] = value
    for k in groups["profiles"]:
        groups["profiles"][k] = w[k].copy()
    summary = GeneratedInputs(
        property_id=request.property_id, n_samples=n_samples, seed=request.seed,
        request_hash=bank.prior.request_hash, facts=copy.deepcopy(bank.prior.facts), prior=bank.prior,
        parameters={k: moments(a, input_unit(k)) for k, a in p.items()},
        constants={k: moments(a, bank.prior.marginals[k].unit) for k, a in constants.items()},
        monthly_inputs={k: [moments(a[:, j], input_unit(k)) for j in range(a.shape[1])] for k, a in w.items()},
        categories={k: {str(v): float(np.mean(a == v)) for v in np.unique(a)}
                    for k, a in groups["latent"].items() if a.dtype.kind in ("U", "S", "O")},
        sampling="posterior_joint_samples" if posterior else "independent_monte_carlo",
        calibration_status="performed" if posterior else "not_performed",
        calibration_id=fit.result.calibration_id if posterior else None,
        calibration_diagnostics={"status": fit.result.status, "method": fit.result.method,
                                 "inherited_constants": list(active), "diagnostics": fit.result.diagnostics}
        if posterior else {},
    )
    return InputSamples(request, bank.prior, p, constants, w, groups["profiles"], [], summary, groups["latent"])


def shared_calibration_result(fit, posterior_ref):
    """Expose the shared Part-1 calibration summary for existing API consumers."""
    from contracts.models import CalibrationDiagnostics, CalibrationResult

    per_parameter = fit.result.diagnostics.get("parameters", {})
    r_hat = [d["r_hat"] for d in per_parameter.values() if d["r_hat"] is not None]
    ess = [d["effective_samples"] for d in per_parameter.values() if d["effective_samples"] is not None]
    return CalibrationResult(
        property_id=fit.result.property_id, method=fit.result.method,
        n_samples=len(next(iter(fit.samples.values()))), months_used=len(fit.result.training_months),
        months_imputed=[m for m in fit.bank.request.months if m not in fit.result.training_months],
        parameters=fit.result.posterior_parameters,
        diagnostics=CalibrationDiagnostics(max_r_hat=max(r_hat) if r_hat else None,
                                           min_ess=min(ess) if ess else None,
                                           divergences=fit.result.diagnostics.get("divergences"),
                                           elbo=-fit.result.diagnostics["final_loss"]
                                           if "final_loss" in fit.result.diagnostics else None),
        data_confidence="medium" if fit.result.status == "converged" else "low",
        posterior_ref=str(posterior_ref),
        prior_source="hierarchy" if fit.bank.prior.application_history or fit.bank.prior.context_audit
        else "rdsap_defaults", provenance=fit.result.provenance,
    )
