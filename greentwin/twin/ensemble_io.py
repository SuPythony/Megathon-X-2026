"""Versioned replay artifacts. Writes are restricted to this worktree."""

import hashlib
import json
from pathlib import Path

import numpy as np

from contracts.uncertain_twin import GeneratedInputs, HouseholdRequest, PriorBundle
from greentwin.twin.missing_inputs import InputSamples
from greentwin.twin.priors import request_hash

ROOT = Path(__file__).resolve().parents[2]


def new_output(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT) or path.exists():
        raise ValueError("Choose a NEW output directory inside PHYSICS MODEL AND BAYESIAN")
    return path


def save_inputs(inputs, output):
    output = new_output(output)
    output.mkdir(parents=True)
    for name, model in (("request", inputs.request), ("priors", inputs.prior), ("inputs", inputs.summary)):
        (output / f"{name}.json").write_text(model.model_dump_json(indent=2) + "\n")
    arrays = {}
    for group in ("parameters", "constants", "weather", "profiles", "latent"):
        arrays.update({f"{group}:{k}": a for k, a in getattr(inputs, group).items()})
    np.savez_compressed(output / "input_samples.npz", **arrays)
    manifest = {
        "version": "prepared-ensemble/0.1.0",
        "property_id": inputs.request.property_id,
        "request_hash": inputs.prior.request_hash,
        "n_samples": inputs.summary.n_samples,
        "seed": inputs.request.seed,
        "calibration_status": inputs.summary.calibration_status,
        "files": {},
    }
    for name in ("request.json", "priors.json", "inputs.json", "input_samples.npz"):
        manifest["files"][name] = hashlib.sha256((output / name).read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    inputs.summary.sample_ref = str(output / "input_samples.npz")
    return output


def load_inputs(path):
    path = Path(path).resolve()
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("version") != "prepared-ensemble/0.1.0":
        raise ValueError("Unsupported prepared input version")
    required = {"request.json", "priors.json", "inputs.json", "input_samples.npz"}
    if set(manifest["files"]) != required:
        raise ValueError("Prepared artifact file set differs")
    for name, digest in manifest["files"].items():
        if hashlib.sha256((path / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Prepared artifact hash mismatch: {name}")
    request = HouseholdRequest.model_validate_json((path / "request.json").read_text())
    prior = PriorBundle.model_validate_json((path / "priors.json").read_text())
    summary = GeneratedInputs.model_validate_json((path / "inputs.json").read_text())
    if request_hash(request) != prior.request_hash or summary.request_hash != prior.request_hash:
        raise ValueError("Prepared artifact request identities differ")
    groups = {k: {} for k in ("parameters", "constants", "weather", "profiles", "latent")}
    with np.load(path / "input_samples.npz", allow_pickle=False) as archive:
        for name in archive.files:
            group, key = name.split(":", 1)
            if group not in groups:
                raise ValueError("Unknown sample group")
            groups[group][key] = archive[name].copy()
    summary.sample_ref = str(path / "input_samples.npz")
    return InputSamples(
        request,
        prior,
        groups["parameters"],
        groups["constants"],
        groups["weather"],
        groups["profiles"],
        [],
        summary,
        groups["latent"],
    )


def save_result(run, output):
    output = new_output(output)
    output.mkdir(parents=True)
    np.savez_compressed(output / "output_samples.npz", **run.samples)
    run.result.sample_ref = str(output / "output_samples.npz")
    (output / "result.json").write_text(run.result.model_dump_json(indent=2) + "\n")
    (output / "metrics.json").write_text(
        json.dumps(
            {
                "elapsed_seconds": run.elapsed_seconds,
                "seed": run.result.seed,
                "data_version": run.result.provenance.data_version,
                "n_samples": run.result.n_samples,
                "calibration_status": run.result.calibration_status,
                "sha256": {
                    name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                    for name in ("result.json", "output_samples.npz")
                },
            },
            indent=2,
        )
        + "\n"
    )
    return output
