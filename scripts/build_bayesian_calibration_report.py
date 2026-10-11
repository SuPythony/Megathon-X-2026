"""Render the measured Bayesian example as separate Markdown, HTML and PDF docs."""

import argparse
import base64
import hashlib
import html
import json
import os
import re
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".report_cache/matplotlib"))

import markdown
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import reportlab
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle


def number(value, places=2):
    return f"{value:,.{places}f}"


def inline(text):
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)
    text = html.escape(text)
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    return re.sub(r"`(.*?)`", r'<font face="TwinVera">\1</font>', text)


def render(report, path):
    body = markdown.markdown(report, extensions=["tables", "fenced_code"])
    for target in re.findall(r'<img[^>]+src="([^"]+)"', body):
        image = (path.parent / target).resolve()
        encoded = base64.b64encode(image.read_bytes()).decode()
        body = body.replace(f'src="{target}"', f'src="data:image/png;base64,{encoded}"')
    css = """body{margin:0;background:#eef4f2;color:#183044;font:16px/1.65 system-ui,sans-serif}
    main{max-width:1180px;margin:32px auto;padding:48px;background:white;border-top:8px solid #0b6e4f}
    h1{font-size:36px;line-height:1.2;color:#0b6e4f}h2{margin-top:42px;border-bottom:1px solid #d9e7e0;padding-bottom:8px;color:#0b6e4f}
    h3{color:#2a6fdb}table{border-collapse:collapse;width:100%;font-size:12px;margin:22px 0}th{background:#0b6e4f;color:white;text-align:left}
    th,td{padding:9px;border-bottom:1px solid #dce8e1}tr:nth-child(even){background:#f3f8f5}pre{background:#133246;color:white;padding:18px;overflow:auto;font-size:12px}
    code{font-size:.88em;overflow-wrap:anywhere}img{width:100%;height:auto}a{color:#2a6fdb}li{margin:8px 0}
    @media(max-width:750px){main{margin:0;padding:20px}table{display:block;overflow:auto}h1{font-size:28px}}
    @media print{body{background:white}main{margin:0;padding:0}h2,h3{break-after:avoid}tr,img{break-inside:avoid}}"""
    path.with_suffix(".html").write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
        'content="width=device-width, initial-scale=1"><title>GreenTwin Bayesian Calibration: Prior to Posterior</title>'
        f"<style>{css}</style></head><body><main>{body}</main></body></html>"
    )
    fonts = Path(reportlab.__file__).parent / "fonts"
    pdfmetrics.registerFont(TTFont("TwinVera", str(fonts / "Vera.ttf")))
    pdfmetrics.registerFont(TTFont("TwinVeraBold", str(fonts / "VeraBd.ttf")))
    pdfmetrics.registerFontFamily("TwinVera", normal="TwinVera", bold="TwinVeraBold",
                                 italic="TwinVera", boldItalic="TwinVeraBold")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TwinBody", fontName="TwinVera", fontSize=8.5, leading=12.5, spaceAfter=7))
    styles.add(ParagraphStyle(name="TwinCell", fontName="TwinVera", fontSize=6.5, leading=8.5))
    for key in ("Title", "Heading1", "Heading2", "Heading3"):
        styles[key].fontName = "TwinVeraBold"
        styles[key].textColor = colors.HexColor("#0b6e4f" if key in ("Title", "Heading1") else "#2a6fdb")
    styles["Title"].fontSize = 23; styles["Title"].leading = 29
    styles["Code"].fontSize = 6.2; styles["Code"].leading = 9
    story, lines, i, width = [], report.splitlines(), 0, 540
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1; continue
        if line.startswith("```"):
            block = []; i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                block.extend(textwrap.wrap(lines[i], width=110, replace_whitespace=False) or [""]); i += 1
            story.extend([Preformatted("\n".join(block), styles["Code"]), Spacer(1, 8)]); i += 1
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"[-: ]+", c) for c in cells):
                    rows.append([Paragraph(inline(c), styles["TwinCell"]) for c in cells])
                i += 1
            count = len(rows[0])
            widths = [140] + [(width - 140) / (count - 1)] * (count - 1) if count >= 5 else [width / count] * count
            table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dceee5")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f8f6")]),
                ("LINEBELOW", (0, 0), (-1, -1), .3, colors.HexColor("#d6e1dc")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.extend([table, Spacer(1, 10)])
        elif line.startswith("!["):
            image = Image(str(path.parent / re.search(r"\((.*?)\)", line).group(1)))
            image.drawHeight = image.imageHeight * width / image.imageWidth; image.drawWidth = width
            story.extend([image, Spacer(1, 10)]); i += 1
        elif line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            style = "Title" if level == 1 else "Heading1" if level == 2 else "Heading2" if level == 3 else "Heading3"
            story.append(Paragraph(inline(line.lstrip("# ")), styles[style])); i += 1
        else:
            paragraph = [line]; i += 1
            if not re.match(r"^(\d+\. |- )", line):
                while i < len(lines) and lines[i].strip() and not re.match(r"^(#|\||```|!\[|- |\d+\. )", lines[i]):
                    paragraph.append(lines[i]); i += 1
            story.append(Paragraph(inline(" ".join(paragraph)), styles["TwinBody"]))

    def footer(canvas, doc):
        canvas.setFont("TwinVera", 7); canvas.setFillColor(colors.HexColor("#4b6575"))
        canvas.drawString(36, 22, "GreenTwin · Bayesian calibration · Controlled synthetic recovery example")
        canvas.drawRightString(576, 22, f"Page {doc.page}")

    SimpleDocTemplate(str(path.with_suffix(".pdf")), pagesize=(612, 792), leftMargin=36, rightMargin=36,
                      topMargin=36, bottomMargin=38, title="GreenTwin Bayesian Calibration: Prior to Posterior",
                      author="Team MegaMind").build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--test-count", type=int, required=True)
    args = parser.parse_args()
    source = args.evidence.resolve().parent
    if not source.is_relative_to(ROOT):
        raise ValueError("Evidence and documentation must stay inside this worktree")
    path = ROOT / "docs/BAYESIAN_CALIBRATION_PRIOR_TO_POSTERIOR_WORKED_EXAMPLE.md"
    if any(path.with_suffix(ext).exists() for ext in (".md", ".html", ".pdf")):
        raise ValueError("Separate documentation already exists")
    e = json.loads(args.evidence.read_text())
    fit = json.loads((source / "calibration/calibration.json").read_text())
    prior_inputs = json.loads((source / "prior_inputs/inputs.json").read_text())
    post_inputs = json.loads((source / "posterior_inputs/inputs.json").read_text())
    results = {name: json.loads((source / name / "result.json").read_text()) for name in ("prior", "posterior", "retrofit")}
    blue, green = "#2a6fdb", "#0b6e4f"
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    x = np.arange(12)
    for name, colour in (("prior", blue), ("posterior", green)):
        ax.plot(x, [r[name]["mean"] for r in e["monthly"]], "o-", color=colour, label=f"{name.title()} mean")
        ax.fill_between(x, [r[name]["percentiles"]["p5"] for r in e["monthly"]],
                        [r[name]["percentiles"]["p95"] for r in e["monthly"]], color=colour, alpha=.12)
    ax.plot(x, [r["true_gas_kwh"] for r in e["monthly"]], "--", color="#253747", label="Known physical truth")
    held = [i for i, r in enumerate(e["monthly"]) if r["split"] == "held_out"]
    ax.scatter(held, [e["monthly"][i]["observed_gas_kwh"] for i in held], marker="*", s=130,
               color="#c58423", zorder=5, label="Held-out meters")
    ax.set_xticks(x, [r["month"][5:] for r in e["monthly"]]); ax.set_xlabel("Month of 2025")
    ax.set_ylabel("Monthly gas, kWh"); ax.set_title("Prior and posterior physics predictions · shaded P5–P95 ranges")
    ax.legend(ncol=2, fontsize=8); fig.savefig(source / "bayesian_monthly_predictions.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    with np.load(source / "calibration/posterior_samples.npz") as draws:
        from contracts.uncertain_twin import PriorMarginal
        from greentwin.twin.missing_inputs import sample_marginal

        for ax, key in zip(axes, e["truth_constants"]):
            spec = PriorMarginal.model_validate(fit["prior_specs"][key])
            initial = sample_marginal(spec, np.random.default_rng(42).uniform(size=4096))
            ax.hist(initial, bins=35, density=True, color=blue, alpha=.4, label="Prior")
            ax.hist(draws[key], bins=30, density=True, color=green, alpha=.6, label="Posterior")
            ax.axvline(e["truth_constants"][key], color="#c58423", linestyle="--", label="Known truth")
            ax.set_xlabel("W/m²K" if key == "wall_u" else "Efficiency"); ax.set_title(key); ax.legend(fontsize=8)
    fig.savefig(source / "bayesian_constant_recovery.png", dpi=180); plt.close(fig)
    reduction = 100 * (1 - e["accuracy"]["held_out"]["posterior"]["mae_vs_meter_kwh"]
                       / e["accuracy"]["held_out"]["prior"]["mae_vs_meter_kwh"])
    annual_errors = {name: 100 * abs(e["outputs"][name]["gas_kwh"]["mean"] - e["truth_gas_annual_kwh"])
                     / e["truth_gas_annual_kwh"] for name in ("prior", "posterior")}
    report = f"""# GreenTwin Bayesian calibration: prior to posterior worked example

**Part 1 · Learn household constants from meters, then rerun the JAX physics twin**  
Team MegaMind · 11 October 2026 · Dedicated worktree: PHYSICS MODEL AND BAYESIAN  
Calibration: `home-calibration/0.1.0` · Physics: `thermal-electricity-fuels/0.3.0`

## 1. What is implemented

The twin now supports the complete requested sequence: initial distributions → Bayesian calibration against household monthly meters → joint posterior constant samples → deterministic physics → final output distributions. The same calibrated household can be used for baseline calculations, what-ifs and retrofit packages.

The implementation uses **NumPyro with the existing JAX physics function inside the likelihood**. NUTS supplies joint MCMC posterior samples and convergence diagnostics. A full-covariance SVI option supplies an explicitly labelled variational approximation. Exact customer facts remain fixed. Missing synthetic inputs remain uncertain and are integrated separately from the inferred constants.

The controlled numerical example demonstrates a **{reduction:.2f}% reduction in held-out monthly gas MAE**. Its annual gas error against the known physical truth falls from **{annual_errors['prior']:.2f}% to {annual_errors['posterior']:.2f}%**. The constants, their distributions, all twelve monthly predictions and the evaluation split are shown below.

**Evidence basis:** this is a controlled synthetic recovery experiment, generated with the same declared physics model plus independent seeded observation noise. It demonstrates calibration and held-out prediction under those conditions. The figures are simulation evidence, rather than a measurement of customer-population accuracy.

## 2. Why calibration changes the output

Before calibration, a building description supplies plausible constants. For example, a wall U-value prior reflects the recorded wall type and eligible population evidence. It does not yet incorporate this household's meter readings. Different plausible constants produce different energy outcomes.

Calibration asks which combinations of constants and unresolved household inputs explain the observed consumption. It updates their probabilities. Final predictions still come from the deterministic physics model, evaluated on those posterior combinations.

`posterior ∝ meter likelihood × initial prior`

Calibration therefore changes the distribution of inputs to the physics engine. It does not overwrite a simulated energy total with the bill or multiply all future predictions by a fitted correction factor. Its output is a joint distribution of plausible physical states, which can also be simulated after a retrofit.

## 3. The statistical model and treatment of uncertainty

Let `theta` contain the selected unknown continuous constants, `z` the remaining jointly generated inputs, and `F(theta, z)` the deterministic JAX twin. For each available monthly meter value:

`meter[m, fuel] ~ StudentT(df=4, location=F(theta,z)[m,fuel], scale=sqrt(5² + (relative_noise × prediction)²))`

The default relative scale is 5% for actual reads and 15% for estimated reads, with a 5 kWh absolute scale floor. The Student-t model is robust to occasional atypical readings. These are scale parameters; for df=4, Student-t SD is scale × √2. Noise settings are configurable and recorded with the fit.

Only supplied training readings contribute likelihood terms. Null readings do not become zero. A supplied zero is an observation. Held-out readings are excluded from inference. The same observation contract supports grid import, mains gas, oil, LPG, solid fuel, district heat, export and PV generation where measurements exist.

The remaining uncertainty uses a frozen bank of jointly generated household realizations. The likelihood is averaged over complete household trajectories:

`log p(meters | theta) ≈ logsumexp(sum_over_observed_months_and_fuels(log p(meter | theta, z[k]))) − log(K)`

Each bank row preserves its geometry, weather, occupancy, demand and equipment dependencies across the whole period. The sum over the observed series happens **before** mixing the rows, preserving one household trajectory across months. This is finite-bank numerical integration, with configurable bank size K.

For final prediction, the code samples `theta` from the posterior and then samples the bank row from `p(z[k] | theta, observed baseline meters)`. It preserves the original half-hour profile seed for that row. Thus unresolved inputs remain uncertain while also reflecting the household's observations. All exact supplied facts stay fixed.

Known absence of PV and a battery permits the identical monthly-demand path inside calibration, because half-hour allocation conserves monthly energy. Installed or uncertain PV/storage uses the complete interval-dispatch path when fitting grid imports. Every final output runs through the full thermal, fuel and half-hour electricity engine.

## 4. Which constants can be calibrated

The default selection considers wall, roof, floor, window and door U-values; ventilation ACH; thermal bridging; heating efficiency/COP; hot-water efficiency/COP; unresolved effective baseload and hot-water demand; and an unknown setpoint. Exact recorded values are excluded automatically.

Additional supported selections include solar gains, retained appliance heat, heating auxiliary power, cylinder-loss multiplier, water temperature rise and battery/inverter performance constants. A parameter must have a nondegenerate initial distribution to be inferred. The existing heat-loss multiplier remains fixed at 1 unless an explicit uncertain prior is supplied; fabric uncertainty is already represented by the individual physical priors.

Calibration preserves dependencies: changing inferred baseload also changes retained internal gains, changing water temperature rise updates water-heating energy, and changing device constants updates the corresponding electrical inputs. Heating and hot-water equipment retain their physical efficiency/COP bounds.

Monthly whole-home meters can constrain combinations of fabric, ventilation and equipment. The joint posterior retains those correlations; it does not assert that every wall or device has been individually measured. Parameters with weak evidence can retain broad distributions.

## 5. Step-by-step execution on a household

1. Validate the household, exact customer facts, requested period and aligned meter observations.
2. Build initial priors using the existing property/context/application-history workflow, or accept an explicitly prepared prior snapshot.
3. Generate and freeze a joint missing-input bank. Calibration itself writes no customer-history observations.
4. Select the requested unknown constants, preserving exact customer and recorded physical values.
5. Construct bounded NumPyro priors and the robust meter likelihood around the JAX twin.
6. Run NUTS or SVI. NUTS adapts during warm-up, then retains posterior draws. Warm-up draws are excluded.
7. Return constant distributions, diagnostics, training/held-out dates, observation counts and provenance.
8. Select joint posterior constant rows and conditional nuisance rows for the desired prediction sample count.
9. Feed those complete realizations to the deterministic physics engine and validate its energy accounting.
10. Return posterior predictive output distributions and joint sample arrays for downstream components.

NUTS checks R-hat, effective sample size, divergences and tree-depth saturation. The default convergence gate requires at least two chains, R-hat ≤ 1.01, effective samples ≥ 50 for each inferred parameter, zero divergences and zero tree-depth-limit hits. A fit failing the gate is labelled `needs_review`; normal posterior prediction is blocked. SVI is labelled `approximate`, with optimization diagnostics and its guide recorded.

## 6. Controlled demonstration: household, meters and split

The SH2 example is a 95 m², two-storey 1970s semi-detached house in BS16 with mains-gas heating. This calibration experiment deliberately fixes all physical quantities except **wall U-value and hot-water efficiency**. That isolates their recovery and makes the numerical comparison interpretable.

| Experiment input | Value |
|---|---|
| Occupants / daytime presence | 4 / part-day |
| Thermostat / heating duration | 20°C / 8 hours per day |
| Known space-heating efficiency | 0.90 |
| Known baseload | 7 kWh/day |
| Known useful hot-water demand | 6 kWh/day |
| Known gas cooking | 1 kWh/day |
| Cylinder / PV / battery | Absent |
| Retained electrical heat fraction | 0.85 |
| Monthly demand multipliers | 1 throughout the year |
| Monthly temperatures, Jan–Dec | 4, 5, 7, 10, 13, 17, 19, 18, 15, 11, 7, 4°C |
| Monthly irradiation, Jan–Dec | 18, 30, 65, 100, 135, 150, 145, 120, 85, 45, 22, 15 kWh/m² |
| Known true wall U-value | 0.85 W/m²K |
| Known true hot-water efficiency | 0.80 |
| Synthetic meter noise | Multiplicative exp(N(0, 0.02²)); seed 731 |
| Training months | January, February, April, May, July, August, October, November |
| Held-out months | March, June, September, December |
| Training observations | 8 gas + 8 electricity = 16 meter values |
| Nuisance bank | 4 rows; other experiment quantities deliberately fixed |
| NUTS settings | 2 chains; 500 warm-up and 1,000 retained draws per chain |
| Total retained joint posterior draws | 2,000 |
| Final physics ensemble | 128 selected realizations per state |

The truth comes from a deterministic run with the declared constants. Noisy monthly readings are generated once. The four held-out months never enter the likelihood. Annual truth is **{number(e['truth_gas_annual_kwh'])} kWh gas**, useful space heating is **{number(e['truth_space_heating_annual_kwh'])} kWh**, and H is **{number(e['truth_heat_loss_coefficient_w_per_k'])} W/K**.

Initial wall U follows a lognormal with arithmetic mean 1.50 and log-SD 0.22. Initial hot-water efficiency follows a scaled beta over 0.1–1.0 with centre 0.84 and concentration 40. The initial summaries below use 4,096 prior draws; posterior summaries use all 2,000 retained joint draws. The output distributions use 128 realizations, so their sample means need not match a plug-in calculation at the parameter means.

## 7. Initial versus final constants

![Initial and calibrated constant distributions](../{source.relative_to(ROOT)}/bayesian_constant_recovery.png)

| Constant | Known truth | Initial mean | Initial SD | Final mean | Final SD | Final P5–P95 |
|---|---:|---:|---:|---:|---:|---:|
"""
    for k, value in e["constants"].items():
        a, b = value["prior"], value["posterior"]
        report += f"| `{k}` | {value['truth']:.4f} | {a['mean']:.4f} | {a['std']:.4f} | {b['mean']:.4f} | {b['std']:.4f} | {b['percentiles']['p5']:.4f}–{b['percentiles']['p95']:.4f} |\n"
    report += "\n| Recovery measure | Wall U-value | Hot-water efficiency |\n|---|---:|---:|\n"
    for label, function in (
        ("Initial absolute error of mean", lambda v: abs(v["prior"]["mean"] - v["truth"])),
        ("Final absolute error of mean", lambda v: abs(v["posterior"]["mean"] - v["truth"])),
        ("Reduction in absolute mean error, %", lambda v: 100 * (1 - abs(v["posterior"]["mean"] - v["truth"])
                                                               / abs(v["prior"]["mean"] - v["truth"]))),
        ("Reduction in parameter SD, %", lambda v: 100 * (1 - v["posterior"]["std"] / v["prior"]["std"])),
    ):
        values = [function(e["constants"][key]) for key in ("wall_u", "hot_water_efficiency")]
        report += f"| {label} | {values[0]:.4f} | {values[1]:.4f} |\n"
    report += """

Both known constants lie inside their final 90% credible intervals. Calibration moves both means toward the known values while preserving uncertainty. A final mean is a summary; actual simulation uses joint posterior samples, rather than setting every household realization to that mean.

### All other constant summaries in the demonstration

These quantities were held fixed to isolate the two inferred constants. The table lists the initial and final means recorded in the prepared input artifacts. It distinguishes an inferred change from a quantity preserved by the experiment.

| Constant | Unit | Initial mean | Final mean | Treatment |
|---|---|---:|---:|---|
"""
    for k, a in prior_inputs["constants"].items():
        b = post_inputs["constants"][k]
        report += f"| `{k}` | {a['unit']} | {a['mean']:.6f} | {b['mean']:.6f} | {'Posterior sample subset' if k in e['truth_constants'] else 'Fixed experiment input'} |\n"
    report += """

## 8. Prior versus posterior physics outputs

![Monthly predictions against known truth and held-out meters](../""" + str(source.relative_to(ROOT)) + """/bayesian_monthly_predictions.png)

| Annual physical output | Prior mean | Prior P5–P95 | Posterior mean | Posterior P5–P95 |
|---|---:|---:|---:|---:|
"""
    for k in ("gas_kwh", "electricity_import_kwh", "space_heating_kwh", "hot_water_kwh", "all_fuel_purchased_kwh"):
        a, b = e["outputs"]["prior"][k], e["outputs"]["posterior"][k]
        report += f"| `{k}` | {number(a['mean'])} | {number(a['percentiles']['p5'])}–{number(a['percentiles']['p95'])} | {number(b['mean'])} | {number(b['percentiles']['p5'])}–{number(b['percentiles']['p95'])} |\n"
    report += "\n| Gas distribution statistic | Prior | Posterior |\n|---|---:|---:|\n"
    for k in ("mean", "variance", "std", "median", "monte_carlo_standard_error"):
        report += f"| `{k}` | {number(e['outputs']['prior']['gas_kwh'][k])} | {number(e['outputs']['posterior']['gas_kwh'][k])} |\n"
    report += f"""

Energy is in kWh; energy variance is in kWh². Electricity and useful hot-water demand stay fixed in this controlled comparison. The inferred wall performance changes useful space heating, and hot-water efficiency changes purchased gas required for the same useful water heat. The prior and posterior mean H values are **{number(results['prior']['metrics']['heat_loss_coefficient_w_per_k']['mean'])} and {number(results['posterior']['metrics']['heat_loss_coefficient_w_per_k']['mean'])} W/K**, against the known **{number(e['truth_heat_loss_coefficient_w_per_k'])} W/K**.

### Held-out numerical comparison

| Measure on four excluded months | Prior | Posterior |
|---|---:|---:|
"""
    for k, label in (("mae_vs_meter_kwh", "Mean absolute error against noisy meters, kWh"),
                     ("rmse_vs_meter_kwh", "RMSE against noisy meters, kWh"),
                     ("mape_vs_meter_pct", "MAPE against noisy meters, %"),
                     ("mae_vs_truth_kwh", "MAE against known physical truth, kWh"),
                     ("physical_range_90_coverage_truth_pct", "Known truth inside physical P5–P95, %")):
        report += f"| {label} | {number(e['accuracy']['held_out']['prior'][k])} | {number(e['accuracy']['held_out']['posterior'][k])} |\n"
    report += f"""

Held-out MAE falls by **{reduction:.2f}%**. This is measured on months excluded from calibration, providing a stronger check than improvement on the fitted bills alone. Four-month coverage is an example count, rather than a population coverage estimate.

### All twelve monthly gas predictions

| Month | Use | Known truth | Noisy meter | Prior mean | Posterior mean | Posterior P5–P95 |
|---|---|---:|---:|---:|---:|---:|
"""
    for row in e["monthly"]:
        d = row["posterior"]
        report += f"| {row['month']} | {row['split']} | {number(row['true_gas_kwh'])} | {number(row['observed_gas_kwh'])} | {number(row['prior']['mean'])} | {number(d['mean'])} | {number(d['percentiles']['p5'])}–{number(d['percentiles']['p95'])} |\n"
    report += """

### Training, held-out and complete-period error

| Split | Prior MAE, kWh | Posterior MAE, kWh | Prior RMSE, kWh | Posterior RMSE, kWh | Prior MAPE, % | Posterior MAPE, % |
|---|---:|---:|---:|---:|---:|---:|
"""
    for split, v in e["accuracy"].items():
        a, b = v["prior"], v["posterior"]
        report += f"| {split} | {number(a['mae_vs_meter_kwh'])} | {number(b['mae_vs_meter_kwh'])} | {number(a['rmse_vs_meter_kwh'])} | {number(b['rmse_vs_meter_kwh'])} | {number(a['mape_vs_meter_pct'])} | {number(b['mape_vs_meter_pct'])} |\n"
    report += """

### Physical output ranges versus meter-observation ranges

The distributions used by downstream parts are **physical energy distributions**, obtained by rerunning the twin. They do not include random bill-reading noise. To check a future noisy meter observation, the example separately adds the declared Student-t observation model to each physical realization.

| Held-out month | Actual synthetic meter | Prior meter-predictive P5–P95 | Posterior meter-predictive P5–P95 |
|---|---:|---:|---:|
"""
    for row in e["monthly"]:
        if row["split"] == "held_out":
            a, b = [e["meter_predictive_90"][name][row["month"]] for name in ("prior", "posterior")]
            report += f"| {row['month']} | {number(row['observed_gas_kwh'])} | {number(a[0])}–{number(a[1])} | {number(b[0])}–{number(b[1])} |\n"
    report += """

## 9. Reusing the posterior for a retrofit or what-if

Calibrate the household once using its actual baseline readings. A hypothetical change then uses that posterior without treating hypothetical consumption as another observation.

`scenario_request(fit, changes)` constructs a proposed state. `predictive_inputs(fit, request=scenario)` selects the same baseline-conditioned latent population as the baseline prediction. Unchanged calibrated components retain their posterior values. An explicit retrofit target replaces the corresponding old value. Replacing heating equipment removes superseded old efficiency facts and uses the proposed technology's initial equipment distribution.

For example, the demonstrated fabric package sets wall U = 0.30, roof U = 0.16 and window U = 1.40 W/m²K. The package replaces the old wall performance while retaining the learned hot-water efficiency. It reruns all edits together.

| Annual quantity | Calibrated baseline mean | Fabric retrofit mean |
|---|---:|---:|
"""
    for k in ("gas_kwh", "electricity_import_kwh", "space_heating_kwh", "all_fuel_purchased_kwh"):
        report += f"| `{k}` | {number(results['posterior']['totals'][k]['mean'])} | {number(results['retrofit']['totals'][k]['mean'])} |\n"
    report += f"""

Matched mean gas saving is **{number(e['retrofit_paired_savings']['gas_kwh']['mean'])} kWh/year**, with P5–P95 **{number(e['retrofit_paired_savings']['gas_kwh']['percentiles']['p5'])}–{number(e['retrofit_paired_savings']['gas_kwh']['percentiles']['p95'])} kWh/year**. Mean electricity saving is **{number(e['retrofit_paired_savings']['electricity_import_kwh']['mean'])} kWh/year** in this fabric-only case. Both results carry the same calibration identity, allowing correlated sample-wise savings calculations.

Changing the building state does not imply that old readings describe the modified building. For a real completed retrofit, subsequently collected readings can support a new calibration revision for the new observed state.

## 10. Final output and downstream handoff

Posterior prediction preserves the existing output structure: 16 period-total distributions, 60 monthly output series, seven building-element contributions and their shares, eight metric slots, complete-year summaries, physical conservation checks and full joint sample arrays.

The final result is labelled `uncertainty_basis = posterior_predictive` and `calibration_status = performed`. It includes `calibration_id` and the method/diagnostic status. SVI results also identify their approximate inference status. These fields distinguish a calibrated result from the earlier prior-only workflow.

Use the posterior physics samples for energy, fuel, solar, storage and element diagnostics. Part 2 can calculate signed per-fuel savings, carbon and bills on matched rows; scoring and value components can consume those results and the property state. Preserve each sample's joint relationships when combining quantities.

`shared_calibration.json` additionally exposes the existing shared `CalibrationResult` contract. The complete prepared posterior inputs retain uncertainty in missing inputs and device specifications, alongside the parameter draws. Passing those prepared inputs to `simulate_ensemble` gives the calibrated full-engine results.

Every summary retains mean, variance, SD, median, mode, percentiles, intervals and histogram. Constant-summary Monte Carlo standard errors from NUTS use diagnostic effective sample size. The existing output-summary MCSE uses the finite prediction-ensemble SD/√n convention; inspect sampler ESS separately when assessing inference precision.

## 11. Running it

### Calibrate against household monthly meters

Save a list of nullable monthly observations such as:

```json
[
  {{"month": "2025-01", "electricity_import_kwh": 300, "gas_kwh": 2400, "estimated": false}},
  {{"month": "2025-02", "electricity_import_kwh": null, "gas_kwh": 2100, "estimated": false}}
]
```

These two rows illustrate the input format. They are not the numerical demonstration's observations. Import the existing `EnergyUsageMonth` records with `observations_from_usage`, or use the nullable meter columns already in Satyarth's physics handoff.

```sh
.venv/bin/python scripts/run_calibrated_twin.py calibrate --request household.json --meters meters.json --output artifacts/household-calibration
.venv/bin/python scripts/run_calibrated_twin.py predict --calibration artifacts/household-calibration --samples 128 --output artifacts/household-posterior
```

Use `--history` during calibration to read an application-history prior source. Use `--config` to select constants and inference settings. The default method is NUTS; set `method` to `svi` for variational inference. Both store the actual chosen configuration.

### Run a proposed scenario on the saved posterior

```sh
.venv/bin/python scripts/run_calibrated_twin.py predict --calibration artifacts/household-calibration --changes retrofit.json --samples 128 --output artifacts/retrofit-posterior
```

For in-process use:

```python
fit = calibrate_household(request, monthly_observations, prior=initial_prior)
baseline_inputs = predictive_inputs(fit, n_samples=128)
baseline_run = simulate_ensemble(baseline_inputs)

proposed = scenario_request(fit, {{"wall_u": 0.30}})
after_run = simulate_ensemble(predictive_inputs(fit, request=proposed, n_samples=128))
per_fuel_savings = paired_differences(baseline_run, after_run)
```

The implementation is in `greentwin/bayes/calibration.py`; import the named functions from that module and the ensemble functions from `greentwin/twin/ensemble.py`. The imports are separated from the abbreviated example above for readability.

Calibration artifacts contain the detailed and shared summaries, the aligned joint posterior sample NPZ, frozen nuisance input artifacts and a hash manifest. Prediction artifacts contain prepared posterior inputs and full result artifacts. Replay verifies hashes and does not refit. New output directories are restricted to this worktree.

## 12. Diagnostics, checks and reproducibility

| Sampler/check | Measured result |
|---|---:|
| Fit status | {e['fit_status']} |
| Wall-U R-hat | {e['diagnostics']['parameters']['wall_u']['r_hat']:.6f} |
| Hot-water-efficiency R-hat | {e['diagnostics']['parameters']['hot_water_efficiency']['r_hat']:.6f} |
| Wall-U effective sample size | {e['diagnostics']['parameters']['wall_u']['effective_samples']:.2f} |
| Hot-water-efficiency effective sample size | {e['diagnostics']['parameters']['hot_water_efficiency']['effective_samples']:.2f} |
| Divergences | {e['diagnostics']['divergences']} |
| Tree-depth-limit hits | {e['diagnostics']['tree_depth_limit_hits']} |
| Mean NUTS acceptance | {e['diagnostics']['mean_acceptance']:.6f} |
| Measured calibration duration | {e['fit_seconds']:.2f} seconds |
| Largest physical accounting residual, prior/posterior/retrofit runs | {max(max(v.values()) for v in e['max_conservation_error_kwh'].values()):.8f} kWh |
| Tests passing across checked physics/input/calibration and shared-contract suites | {args.test_count} |
| New calibration tests | 12 |

The twelve calibration tests cover held-out isolation, missing versus zero readings, observation alignment, exact-fact preservation, dependent gains, NUTS recovery, posterior replay and tamper detection, posterior retrofit pairing, diagnostic gating, Satyarth handoff, SVI, conditional nuisance sampling, equipment replacement and differentiable PV/battery grid-import likelihood.

The measured fit duration describes this controlled two-constant example and its settings. It is not a latency guarantee for other household sizes, bank sizes or installed equipment.

Reproduce the experiment in a new directory:

```sh
.venv/bin/python scripts/evaluate_bayesian_calibration.py --output eval/reports/new-bayesian-example --warmup 500 --samples 1000 --predictive-samples 128
```

Evidence SHA-256: `{hashlib.sha256(args.evidence.read_bytes()).hexdigest()}`.

"""
    links = [
        ("Full numerical evidence", "evidence.json"), ("Example household", "request.json"),
        ("All noisy meters with training flags", "meters.json"), ("Initial priors and frozen nuisance bank", "calibration/nuisance_inputs/priors.json"),
        ("Detailed calibration and before/after constants", "calibration/calibration.json"),
        ("Shared calibration contract", "calibration/shared_calibration.json"),
        ("All 2,000 joint posterior constant draws", "calibration/posterior_samples.npz"),
        ("Prior physical outputs", "prior/result.json"), ("Posterior physical outputs", "posterior/result.json"),
        ("Prepared posterior input distributions", "posterior_inputs/inputs.json"),
        ("Joint posterior physics output draws", "posterior/output_samples.npz"),
        ("Posterior-based retrofit result", "retrofit/result.json"),
    ]
    for label, name in links:
        report += f"- [{label}](../{source.relative_to(ROOT)}/{name})\n"
    report += "\nMethod references: [NumPyro MCMC and NUTS](https://num.pyro.ai/en/latest/mcmc.html), [NumPyro SVI](https://num.pyro.ai/en/latest/svi.html).\n"
    for target in re.findall(r"\]\(([^)]+)\)", report):
        if not target.startswith("https:") and not (path.parent / target).exists():
            raise ValueError(f"Broken documentation reference: {target}")
    path.write_text(report)
    render(report, path)
    print(f"Created {path.name}, HTML and PDF; {len(report.split())} words")


if __name__ == "__main__":
    main()
