"""Hash-verified calibration artifacts, with worktree-only writes."""

import hashlib
import json
from pathlib import Path

import numpy as np

from contracts.calibration import HomeCalibrationResult
from greentwin.bayes.calibration import CalibrationFit, CalibrationModel, shared_calibration_result
from greentwin.twin.ensemble_io import load_inputs, new_output, save_inputs


def save_calibration(fit, output):
    output = new_output(output)
    output.mkdir(parents=True)
    save_inputs(fit.bank, output / "nuisance_inputs")
    np.savez_compressed(output / "posterior_samples.npz", **fit.samples)
    (output / "calibration.json").write_text(fit.result.model_dump_json(indent=2) + "\n")
    (output / "shared_calibration.json").write_text(shared_calibration_result(
        fit, output / "posterior_samples.npz"
    ).model_dump_json(indent=2) + "\n")
    names = ["calibration.json", "shared_calibration.json", "posterior_samples.npz", "nuisance_inputs/manifest.json"]
    (output / "manifest.json").write_text(json.dumps({
        "version": "calibration-artifact/0.1.0", "calibration_id": fit.result.calibration_id,
        "files": {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in names},
    }, indent=2) + "\n")
    return output


def load_calibration(path):
    path = Path(path).resolve()
    manifest = json.loads((path / "manifest.json").read_text())
    names = {"calibration.json", "shared_calibration.json", "posterior_samples.npz", "nuisance_inputs/manifest.json"}
    if manifest.get("version") != "calibration-artifact/0.1.0" or set(manifest["files"]) != names:
        raise ValueError("Unsupported calibration artifact or file set")
    for name, digest in manifest["files"].items():
        if hashlib.sha256((path / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Calibration artifact hash mismatch: {name}")
    result = HomeCalibrationResult.model_validate_json((path / "calibration.json").read_text())
    if result.calibration_id != manifest["calibration_id"]:
        raise ValueError("Calibration identities differ")
    bank = load_inputs(path / "nuisance_inputs")
    if bank.prior.request_hash != result.request_hash or bank.request.property_id != result.property_id:
        raise ValueError("Calibration and nuisance inputs belong to different requests")
    with np.load(path / "posterior_samples.npz", allow_pickle=False) as archive:
        samples = {k: archive[k].copy() for k in archive.files}
    count = result.config.samples * result.config.chains
    if set(samples) != set(result.fitted_parameters) or any(
        a.shape != (count,) or not np.isfinite(a).all() for a in samples.values()
    ):
        raise ValueError("Invalid posterior sample population")
    model = CalibrationModel(bank, result.observations, result.config, result.prior_specs)
    return CalibrationFit(result, bank, samples, model)
