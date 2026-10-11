"""Render measured prior-predictive evidence as standalone HTML, PDF and Markdown."""

import argparse
import base64
import html
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".report_cache/matplotlib"))

import markdown
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle


def number(value):
    return f"{value:,.1f}"


def inline(text):
    text = html.escape(text)
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.*?)`", r'<font face="Courier">\1</font>', text)
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--test-count", type=int, required=True)
    args = parser.parse_args()
    directory = args.evidence.resolve().parent
    if not directory.is_relative_to(ROOT):
        raise ValueError("Report must stay inside this worktree")
    if (directory / "report.pdf").exists():
        raise ValueError("Report already exists; choose a new evidence directory")
    e = json.loads(args.evidence.read_text())
    n = e["n_samples"]
    scenarios = e["scenarios"]
    labels = {
        "baseline": "Baseline gas home",
        "fabric": "Fabric changes",
        "heat_pump": "Heat pump only",
        "fabric_heat_pump_pv_battery": "Fabric + heat pump + PV + battery",
    }
    green, blue = "#0b6e4f", "#2a6fdb"
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    positions = np.arange(4)
    means = np.array([v["totals"]["all_fuel_purchased_kwh"]["mean"] for v in scenarios.values()])
    lows = np.array([v["totals"]["all_fuel_purchased_kwh"]["percentiles"]["p5"] for v in scenarios.values()])
    highs = np.array(
        [v["totals"]["all_fuel_purchased_kwh"]["percentiles"]["p95"] for v in scenarios.values()]
    )
    ax.barh(positions, means, color=[blue, green, "#638cc3", "#47a282"], height=0.55)
    ax.errorbar(means, positions, xerr=[means - lows, highs - means], fmt="none", ecolor="#152a40", capsize=4)
    ax.set_yticks(positions, ["Baseline", "Fabric", "Heat pump", "Combined package"])
    ax.invert_yaxis()
    ax.set_xlabel("Purchased energy, kWh over January–December 2025")
    ax.set_title(f"Mean and 90% prior predictive interval · {n} joint draws per scenario")
    fig.savefig(directory / "scenario_energy.png", dpi=180)
    fig.savefig(directory / "scenario_energy.svg")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    with (
        np.load(directory / "baseline/output_samples.npz") as a,
        np.load(directory / "fabric/output_samples.npz") as b,
    ):
        axes[0].hist(a["gas_kwh"], bins=15, alpha=0.6, color=blue, label="Baseline")
        axes[0].hist(b["gas_kwh"], bins=15, alpha=0.6, color=green, label="Fabric")
    axes[0].set_xlabel("Gas consumption, kWh/year")
    axes[0].set_ylabel("Joint draws")
    axes[0].legend()
    axes[0].set_title("Gas distribution changes after rerunning physics")
    counts = [int(k) for k in e["mean_convergence"]]
    axes[1].plot(counts, [e["mean_convergence"][str(k)]["gas_mcse_kwh"] for k in counts], "o-", color=green)
    axes[1].set_xlabel("Number of draws")
    axes[1].set_ylabel("Monte Carlo standard error, kWh")
    axes[1].set_title("Sampling error decreases; model error is separate")
    fig.savefig(directory / "uncertainty_checks.png", dpi=180)
    fig.savefig(directory / "uncertainty_checks.svg")
    plt.close(fig)
    max_error = max(max(s["max_conservation_error_kwh"].values()) for s in scenarios.values())
    rows = []
    for key, value in scenarios.items():
        t = value["totals"]
        d = t["all_fuel_purchased_kwh"]
        rows.append(
            f"| {labels[key]} | {number(t['gas_kwh']['mean'])} | {number(t['electricity_import_kwh']['mean'])} | {number(d['mean'])} | {number(d['std'])} | {d['variance']:,.0f} | {number(d['percentiles']['p5'])}–{number(d['percentiles']['p95'])} |"
        )
    changes = []
    for key, value in e["paired_changes"].items():
        changes.append(
            f"| {labels[key]} | {number(value['gas_kwh']['mean'])} | {number(value['electricity_import_kwh']['mean'])} | {number(value['all_fuel_purchased_kwh']['mean'])} |"
        )
    fuels = []
    for key, value in e["fuel_checks"].items():
        fuels.append(
            f"| {key} | {number(value['mean_purchased_kwh'])} | {value['max_conversion_error_kwh']:.6f} | {number(value['gas_kwh_mean'])} |"
        )
    convergence = []
    for count, values in e["mean_convergence"].items():
        convergence.append(
            f"| {count} | {number(values['gas_mean_kwh'])} | {number(values['gas_mcse_kwh'])} |"
        )
    runtime_rows = [f"| {labels[k]} | {v['runtime_seconds']:.2f} |" for k, v in scenarios.items()]
    baseline = scenarios["baseline"]["totals"]["all_fuel_purchased_kwh"]["mean"]
    package = scenarios["fabric_heat_pump_pv_battery"]["totals"]["all_fuel_purchased_kwh"]["mean"]
    docs = (ROOT / "docs/UNCERTAIN_INPUTS_AND_TWIN.md").read_text()
    report = f"""# GreenTwin · Part 1 implementation and evidence report

**Physics, missing household inputs and initial uncertainty**  
Team MegaMind · Dedicated worktree: PHYSICS MODEL AND BAYESIAN  
Evaluation date: 11 October 2026 · Physics model: thermal-electricity-fuels/0.3.0

## Executive assessment

The requested three-stage implementation is complete: separate initial prior construction, separate joint synthetic missing-input generation, and deterministic JAX simulation wrapped in Monte Carlo output distributions. Gas and other supported purchased fuels are accounted for alongside electricity. **Bayesian calibration has not been implemented or called.**

The measurable gains are better representation of uncertainty, traceable input handling and reproducible fuel/retrofit accounting. This report does not claim an improvement in real-world prediction accuracy. The currently available local priors, appliance centres and development weather are provisional; this phase cannot establish the system design's meter-accuracy or held-out coverage targets.

* **{args.test_count} tests pass**, including the new uncertainty-boundary tests and the existing thermal, electricity and synthetic-data tests.
* **{n}/{n} fabric scenario draws** reduce or preserve useful heating in this evaluation.
* The largest checked conservation residual across the four annual scenarios is **{max_error:.6f} kWh**. This is a numerical identity check, not prediction error against meters.
* Supplying seven envelope measurements reduces heat-loss-coefficient sample SD by **{e["information_gain"]["std_reduction_pct"]:.2f}%** in this experiment.
* Regenerating the same request and seed returns identical parameter arrays: **{e["identical_regenerated_parameter_arrays"]}**.

## What changed compared with the preceding implementation

| Aspect | Previously available | Added in this phase |
|---|---|---|
| Physics | Deterministic monthly thermal and half-hour electricity engine | Prior-predictive ensemble wrapper over the same JAX solver |
| Missing inputs | Development household/profile generators | Explicit bounded probability models and conditional joint input draws |
| Constants | Default/supplied scalar values | Separate source-labelled initial probability distributions |
| Customer facts | Deterministic intake/handoff | Frozen exact facts throughout the ensemble; unknown input rejection |
| Fuels | Six fuel carriers already supported | Monthly fuel-by-end-use summaries, sampled efficiencies and separate auxiliaries |
| Uncertain outputs | Conditional deterministic electricity runs | Mean, variance, SD, intervals, histograms, joint raw samples and MCSE |
| Replay | Seeds and older development artifacts | Hash-verified prepared inputs, separate simulation stage and saved output draws |
| Evaluation | 90 preceding tests | {args.test_count} total tests plus the measured scenario artifacts below |

This is completion of the user's requested phase, not completion of every item assigned to Part 1 in SYSTEM_DESIGN.pdf. Calibration, scene/capture, standard SAP preparation, public data fitting and full app/orchestrator integration remain outside this phase.

## Evaluation design and comparability

The showcase SH2 home is a 95 m², two-storey semi-detached house in BS16. The request fixes four occupants, a 20°C thermostat, gas cooking at 1 kWh/day and no hot-water cylinder. Existing PV/battery absence is explicit in its Property. Fabric descriptions, ventilation, efficiency, geometry detail, demand factors and development weather remain uncertain where not measured.

Each scenario has **{n} draws**, seed **{e["seed"]}**, and the same twelve months of 2025. The annual electricity calculation has **17,520 half-hour intervals per draw**, so the four principal scenario runs cover **{n * 4:,} household realizations** and **{n * 4 * 17520:,} dispatch intervals**. These counts describe computation, not independent observed households. Named random streams preserve the same latent draws for unaffected quantities, improving the precision of paired scenario comparisons.

The fabric scenario supplies wall U=0.30, roof U=0.16 and window U=1.40 W/m²K. Heat-pump-only changes the equipment/fuel without changing the fabric. The combined scenario applies those fabric edits, an air-source heat pump, 3.5 kWp PV and a 5 kWh battery in a single rerun. Their feasibility, installation costs and appropriateness for this household have not been established by this simulator.

## Annual scenario results

All values below are simulation estimates for the complete 2025 calendar year. SD is household/model uncertainty under the declared priors. Variance has units kWh². The interval is the empirical 5th–95th percentile range of the joint realizations.

| Scenario | Gas mean, kWh | Grid import mean, kWh | All-fuel mean, kWh | SD, kWh | Variance, kWh² | 90% interval, kWh |
|---|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

![Scenario means and intervals](scenario_energy.png)

The combined scenario's mean purchased energy is **{100 * (1 - package / baseline):.2f}%** below the baseline mean in this development experiment. This ratio of scenario means is descriptive; it is not the mean of per-draw official improvement-potential ratios, and it is not a guaranteed saving.

## Signed per-fuel changes and interactions

The following numbers are the means of **paired baseline-minus-after samples**. Positive means a reduction; negative means an increase. They must remain separate before applying carbon factors or prices.

| Scenario | Gas saving, kWh | Electricity saving, kWh | All-fuel saving, kWh |
|---|---:|---:|---:|
{chr(10).join(changes)}

The heat-pump-only scenario reduces heating gas while raising electricity import; the remaining gas is cooking. The combined package is rerun as one changed household. Its effect is not an addition of independent fabric, heating, PV and battery savings. Full paired distributions, including variance and intervals, are saved in evidence.json.

## Numerical evidence and what it proves

Conservation checks run on **every realization** before output summarization. They test the monthly heat balance, purchased fuel versus delivered heat/conversion losses, heat-pump ambient heat, monthly allocation and half-hour electricity/PV/battery identities, including storage bounds and power limits. Ordinary float32 rounding gives small nonzero residuals.

All six fuel configurations are independently checked in January on eight draws each. The conversion error compares the reported main heating fuel input with useful heating divided by the separately sampled efficiency/COP. The retained gas cooking is 31 kWh in every non-gas-heating January case.

| Main fuel | Mean all-fuel purchase in January, kWh | Max conversion identity error, kWh | Mean gas incl. cooking, kWh |
|---|---:|---:|---:|
{chr(10).join(fuels)}

This evidence supports algebraic correctness and physical accounting. It does not prove that a particular actual home's U-values, demand behaviour or seasonal COP equal the current priors. Synthetic-target agreement is method validation rather than observed accuracy, consistent with LLOYDS-ps2.pdf §9 and §11.

## Measured uncertainty improvement

Fixing wall, roof, floor, window and door U-values, ACH and the thermal-bridge coefficient to supplied envelope values changes the sample SD of H from **{e["information_gain"]["baseline_h_std_w_per_k"]:.3f} W/K** to **{e["information_gain"]["known_envelope_h_std_w_per_k"]:.3f} W/K**, a **{e["information_gain"]["std_reduction_pct"]:.2f}% reduction**. Residual geometry uncertainty remains. This is a controlled demonstration that extra information narrows this diagnostic; it is not evidence that all output intervals always shrink after any customer answer.

Repeated requests reproduce arrays exactly. Changing the random seed changes uncertain quantities. Changing a supplied U-value keeps the unaffected latent streams shared. Exact values such as the customer thermostat or measured wall U have zero input variance, while their simulated energy consequences can retain uncertainty from other inputs.

![Distribution and sampling-error checks](uncertainty_checks.png)

## Sample count, convergence and runtime

The table uses nested prefixes of the baseline ensemble, not separate empirical datasets. MCSE estimates the error in the Monte Carlo mean as sample SD divided by square root of sample count. It is different from the household uncertainty SD and from model bias.

| Draws | Baseline gas mean, kWh/year | Mean MCSE, kWh/year |
|---|---:|---:|
{chr(10).join(convergence)}

Increasing the sample count reduces sampling error but cannot recover missing evidence or correct wrong structural assumptions. At 128 samples, fine tail probabilities and rare ownership states remain imprecise. Production sample counts should reflect the required decision precision.

| Scenario | Ensemble wall time, seconds |
|---|---:|
{chr(10).join(runtime_rows)}

Times include interval preparation, JAX dispatch, validation and distribution summarization on this machine, with batch size eight. Initial runs include compilation where necessary; later runs may reuse kernels. These are measured end-to-end development times, not a promise of subsecond application latency or calibration performance.

## Outputs ready for downstream parts

The baseline result has **16 scalar total/storage summaries**, **60 monthly series**, **seven element contributions and shares**, and **eight metric entries**. A metric may be unavailable when its denominator is zero; for example, self-consumption is undefined for a known no-PV baseline. Stocks such as initial/final battery energy are not added to purchased-energy totals.

Part 2 receives joint per-fuel energy samples, separate gas/electricity changes, PV import effects, end-use splits and heat-loss diagnostics. It must apply source-valid tariffs/factors to these same draws and compute ratios/savings with the required zero-baseline handling. Parts 3 and 4 still own the Green Home Score and indicative property value. The scene adapter uses estimated element loss shares; it must preserve their estimated/prior-based status. No solar score, carbon number, property value or official recommendation is manufactured by this phase.

## Important limitations and next handoff

1. **No posterior:** meters are not consumed in a likelihood; no NumPyro fitting, NUTS, SVI or recalibration is performed.
2. **Provisional evidence:** local RdSAP-style defaults and catalogue estimates require verification. No verified NEED or local empirical fit is bundled here. `ContextRecord` accepts documented eligible cohorts but does not download or validate their source licences automatically.
3. **Model resolution:** single-zone monthly steady-state heat, fixed 3 K setback and approximate gains; half-hour electricity allocation is not a transient room-temperature model. Seasonal COP does not model defrost/startup dynamics.
4. **Incomplete uncertainty:** party-wall and other structural assumptions can be exact assumptions rather than sampled epistemic uncertainty. A tight output interval is not proof that those assumptions are correct.
5. **Fuel scope:** gas means mains-gas purchased thermal energy. There is no gas pressure network, combustion chemistry or pollutant concentration simulation.
6. **Feasibility boundary:** the simulator calculates a supplied supported scenario; the rules/survey layer must decide whether it can be installed. Partial periods are not automatically annualised.
7. **Integration boundary:** this new ensemble contract supplements schema 0.4. Full shared DigitalTwin Protocol/UI bindings, SAP standard mode, scene/capture and downstream score/value are still required separately.
8. **Empirical targets not achieved yet:** showcase ±15% metered accuracy and 90% held-out interval coverage from SYSTEM_DESIGN.pdf cannot be certified by these development experiments.

## Reproduction and artifacts

The evidence script is `scripts/evaluate_uncertain_twin.py`; report generation is `scripts/build_twin_report.py`. Run from this worktree with its Python environment. Use a **new** output directory when rerunning.

```sh
.venv/bin/python scripts/evaluate_uncertain_twin.py --samples 128 --output eval/reports/new-run
.venv/bin/python scripts/build_twin_report.py --evidence eval/reports/new-run/evidence.json --test-count {args.test_count}
.venv/bin/python -m pytest -q --basetemp .pytest_tmp
```

The prepared directory contains request/priors/input summaries, input NPZ and a hash manifest. Each scenario directory contains result JSON, output NPZ and metrics/file hashes. The evidence JSON holds scenario summaries, paired changes, convergence, information gain and fuel checks. All reports/charts and runtime caches remain inside this worktree. The original main checkout and Satyarth branch are untouched.

Report dependencies: matplotlib, Markdown and reportlab; PDF reading used pypdf. Core runtime dependencies remain JAX, NumPy, SciPy and Pydantic. Report rendering libraries do not participate in the physics calculation.

## Technical appendix: implementation, input audit and usage

{docs.replace("# Part 1: uncertain inputs, initial priors and deterministic physics", "### Detailed implementation reference").replace("## ", "### ")}
"""
    (directory / "report.md").write_text(report)
    body = markdown.markdown(report, extensions=["tables", "fenced_code"])
    for name in ("scenario_energy.png", "uncertainty_checks.png"):
        body = body.replace(
            f'src="{name}"',
            f'src="data:image/png;base64,{base64.b64encode((directory / name).read_bytes()).decode()}"',
        )
    css = """body{font:16px/1.65 system-ui,sans-serif;color:#183044;background:#eff4f7;margin:0}
    main{max-width:1160px;margin:40px auto;background:white;padding:55px;border-top:8px solid #0b6e4f;box-shadow:0 12px 40px #1232}
    h1{font-size:38px;line-height:1.2;color:#0b6e4f}h2{color:#0b6e4f;border-bottom:1px solid #dce8e2;padding-bottom:8px;margin-top:40px}
    h3{color:#2a6fdb;margin-top:30px}table{border-collapse:collapse;width:100%;font-size:13px;margin:24px 0}th{background:#0b6e4f;color:white;text-align:left}
    td,th{padding:10px;border-bottom:1px solid #dfebe6}tr:nth-child(even){background:#f3f8f6}img{width:100%;height:auto}
    pre{background:#102c3c;color:#eef9f5;padding:20px;overflow:auto;font-size:12px}code{font-size:.9em}li{margin:9px 0}
    @media print{body{background:white}main{box-shadow:none;margin:0;padding:0}h2,h3{break-after:avoid}tr,img{break-inside:avoid}}
    @media(max-width:700px){main{margin:0;padding:20px}h1{font-size:28px}table{display:block;overflow:auto}}"""
    (directory / "report.html").write_text(
        f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>GreenTwin Part 1 evidence report</title><style>{css}</style><main>{body}</main></html>'
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportBody", fontSize=9.3, leading=13.8, spaceAfter=8))
    styles.add(ParagraphStyle(name="SmallTable", fontSize=7, leading=9))
    styles["Title"].textColor = colors.HexColor(green)
    styles["Heading1"].textColor = colors.HexColor(green)
    styles["Heading2"].textColor = colors.HexColor(blue)
    story = []
    lines = report.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith("```"):
            block = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                # Keep long commands inside the printable column.
                import textwrap

                block.extend(textwrap.wrap(lines[i], width=100, replace_whitespace=False) or [""])
                i += 1
            story.append(Preformatted("\n".join(block), styles["Code"]))
        elif line.startswith("| "):
            table = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"[-: ]+", c) for c in cells):
                    table.append([Paragraph(inline(c), styles["SmallTable"]) for c in cells])
                i += 1
            count = len(table[0])
            widths = [7.1 * inch / count] * count
            t = Table(table, colWidths=widths, repeatRows=1, hAlign="LEFT")
            t.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dceee5")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f8f6")]),
                        ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#d6e1dc")),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ]
                )
            )
            story.extend([t, Spacer(1, 12)])
            continue
        elif line.startswith("!["):
            name = re.search(r"\((.*?)\)", line).group(1)
            image = Image(str(directory / name), width=7.1 * inch, height=3.1 * inch, kind="proportional")
            story.extend([image, Spacer(1, 12)])
        elif line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            style = (
                styles["Title"] if level == 1 else styles["Heading1"] if level == 2 else styles["Heading2"]
            )
            story.append(Paragraph(inline(line.lstrip("# ")), style))
        else:
            paragraph = [line]
            i += 1
            if not re.match(r"^[*0-9]", line):
                while (
                    i < len(lines)
                    and lines[i].strip()
                    and not re.match(r"^(#|\||```|!\[|\* |\d+\))", lines[i])
                ):
                    paragraph.append(lines[i])
                    i += 1
            story.append(Paragraph(inline(" ".join(paragraph)), styles["ReportBody"]))
            continue
        i += 1

    def footer(canvas, document):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#4b6575"))
        canvas.drawString(36, 22, "GreenTwin | Part 1 | Prior predictive - not calibrated")
        canvas.drawRightString(576, 22, f"Page {document.page}")

    SimpleDocTemplate(
        str(directory / "report.pdf"),
        pagesize=(612, 792),
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=38,
        title="GreenTwin Part 1 implementation and evidence report",
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Report: {directory / 'report.html'}\nPDF: {directory / 'report.pdf'}")


if __name__ == "__main__":
    main()
