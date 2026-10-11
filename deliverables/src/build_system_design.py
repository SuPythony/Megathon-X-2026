"""Build SYSTEM_DESIGN.pdf (repo root): GreenTwin system design around the five parts.

Dataset tables come from deliverables/src/datasets.py (shared with the LLD).

Run:  python deliverables/src/build_system_design.py
Needs reportlab. Uses Segoe UI (Windows) or DejaVu Sans (Linux), else Helvetica.
"""

import math
import sys
from pathlib import Path

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    CondPageBreak, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "SYSTEM_DESIGN.pdf"


# ----------------------------------------------------------------------------- fonts
def _fonts():
    for d, reg, bold, ital in [
        ("C:/Windows/Fonts", "segoeui.ttf", "segoeuib.ttf", "segoeuii.ttf"),
        ("/usr/share/fonts/truetype/dejavu", "DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans-Oblique.ttf"),
    ]:
        try:
            pdfmetrics.registerFont(TTFont("GT", str(Path(d) / reg)))
            pdfmetrics.registerFont(TTFont("GTB", str(Path(d) / bold)))
            pdfmetrics.registerFont(TTFont("GTI", str(Path(d) / ital)))
            pdfmetrics.registerFontFamily("GT", normal="GT", bold="GTB", italic="GTI", boldItalic="GTB")
            return "GT", "GTB"
        except Exception:
            continue
    return "Helvetica", "Helvetica-Bold"


F, FB = _fonts()

GREEN = colors.HexColor("#0b6e4f")
GREEN_L = colors.HexColor("#e6f2ec")
BLUE = colors.HexColor("#2a6fdb")
BLUE_L = colors.HexColor("#e7effc")
AMBER_L = colors.HexColor("#fdf3e1")
GREY = colors.HexColor("#5b5b5b")
GRID = colors.HexColor("#c9c9c9")
INK = colors.HexColor("#1d1d1d")
LINE = colors.HexColor("#8a8a8a")
PANEL = colors.HexColor("#f4f4f0")

S = {
    "title": ParagraphStyle("title", fontName=FB, fontSize=28, leading=34, textColor=GREEN),
    "subtitle": ParagraphStyle("subtitle", fontName=F, fontSize=13.5, leading=19, textColor=INK),
    "meta": ParagraphStyle("meta", fontName=F, fontSize=10, leading=14, textColor=GREY),
    "h1": ParagraphStyle("h1", fontName=FB, fontSize=15.5, leading=20, textColor=GREEN, spaceBefore=4, spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName=FB, fontSize=11.2, leading=15, textColor=INK, spaceBefore=7, spaceAfter=3),
    "body": ParagraphStyle("body", fontName=F, fontSize=9.3, leading=13.2, textColor=INK, spaceAfter=4),
    "bullet": ParagraphStyle("bullet", fontName=F, fontSize=9.2, leading=12.9, leftIndent=12, bulletIndent=2,
                             spaceAfter=1.5, textColor=INK),
    "cell": ParagraphStyle("cell", fontName=F, fontSize=8.1, leading=10.6, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName=FB, fontSize=8.1, leading=10.6, textColor=INK),
    "small": ParagraphStyle("small", fontName=F, fontSize=7.8, leading=10.2, textColor=GREY),
    "callout": ParagraphStyle("callout", fontName=F, fontSize=9.5, leading=13.6, textColor=INK),
}

story: list = []
W_TEXT = 170 * mm


def P(text, style="body"):
    story.append(Paragraph(text, S[style]))


def H1(text):
    story.append(Paragraph(text, S["h1"]))


def H2(text):
    story.append(Paragraph(text, S["h2"]))


def bullets(items, style="bullet"):
    for it in items:
        story.append(Paragraph(it, S[style], bulletText="\u2022"))


def table(rows, widths, header=True, zebra=True, first_col_bold=False):
    data = []
    for i, r in enumerate(rows):
        row = []
        for j, c in enumerate(r):
            st = S["cellb"] if (header and i == 0) or (first_col_bold and j == 0) else S["cell"]
            row.append(Paragraph(str(c), st))
        data.append(row)
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [
        ("GRID", (0, 0), (-1, -1), 0.4, GRID),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), GREEN_L))
    if zebra:
        for i in range(1, len(rows)):
            if i % 2 == 0:
                st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f8f8f8")))
    t.setStyle(TableStyle(st))
    story.append(t)
    story.append(Spacer(1, 6))


def callout(text, fill=GREEN_L, bar=GREEN):
    t = Table([[Paragraph(text, S["callout"])]], colWidths=[W_TEXT])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), fill),
        ("LINEBEFORE", (0, 0), (0, -1), 3, bar),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))


# ----------------------------------------------------------------------------- drawing helpers
def box(d, x, y, w, h, title, lines=(), kind="plain", tsize=8.2, lsize=6.9):
    stroke, fill, sw = {
        "plain": (colors.HexColor("#9a9a9a"), colors.white, 0.9),
        "core": (BLUE, BLUE_L, 1.6),
        "green": (GREEN, GREEN_L, 1.2),
        "amber": (colors.HexColor("#c8902a"), AMBER_L, 1.0),
        "panel": (colors.HexColor("#9a9a9a"), PANEL, 0.9),
    }[kind]
    d.add(Rect(x, y, w, h, rx=5, ry=5, strokeColor=stroke, strokeWidth=sw, fillColor=fill))
    cy = y + h - 12
    d.add(String(x + w / 2, cy, title, fontName=FB, fontSize=tsize, fillColor=INK, textAnchor="middle"))
    for i, ln in enumerate(lines):
        d.add(String(x + w / 2, cy - 10.5 - i * 9, ln, fontName=F, fontSize=lsize, fillColor=GREY,
                     textAnchor="middle"))


def arrow(d, x1, y1, x2, y2, color=LINE, width=0.9, head=5):
    d.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=width))
    a = math.atan2(y2 - y1, x2 - x1)
    hx1 = x2 - head * math.cos(a) + 3 * math.sin(a)
    hy1 = y2 - head * math.sin(a) - 3 * math.cos(a)
    hx2 = x2 - head * math.cos(a) - 3 * math.sin(a)
    hy2 = y2 - head * math.sin(a) + 3 * math.cos(a)
    d.add(Polygon([x2, y2, hx1, hy1, hx2, hy2], fillColor=color, strokeColor=None))


def label(d, x, y, text, size=6.4, color=GREY, anchor="middle", bold=False):
    d.add(String(x, y, text, fontName=FB if bold else F, fontSize=size, fillColor=color, textAnchor=anchor))




# ----------------------------------------------------------------------------- diagram 1: the five parts
def parts_overview():
    W, H = 482, 430
    d = Drawing(W, H)
    # Part 0
    box(d, 0, 384, W, 40, "Part 0 · Data layer",
        ["EPC records + live tables · NEED · DESNZ LSOA · UCL sales + HPI · weather · smart meters · geography"],
        kind="green", tsize=8.6)
    # coupled core outline
    d.add(Rect(-4, 216, 324, 146, rx=7, ry=7, strokeColor=BLUE, strokeWidth=0.8, strokeDashArray=[3, 2],
               fillColor=None))
    label(d, 158, 222, "COUPLED CORE (one team, sequential)", 6.2, BLUE, "middle", True)
    box(d, 0, 236, 135, 118, "Part 1 · Digital twin",
        ["physics (JAX, RC model)", "Bayesian calibration", "(NumPyro, posterior samples)", "3D scene for the viewer",
         "photo-to-twin capture", "electricity layer"], kind="core", tsize=8.2, lsize=6.6)
    box(d, 180, 236, 135, 118, "Part 2 · Energy & retrofit",
        ["EBM energy prior (NEED)", "baseline per fuel", "improvement ladder", "savings · carbon · bills",
         "improvement potential", "rating after · Pareto"], kind="core", tsize=8.2, lsize=6.6)
    box(d, 350, 298, 132, 56, "Part 3 · Green Home Score",
        ["fixed weights, monotonic", "waterfall · P(band) · issues"], tsize=7.8, lsize=6.5)
    box(d, 350, 236, 132, 56, "Part 4 · Property value",
        ["Bayesian hierarchical", "hedonic · indicative range"], tsize=7.8, lsize=6.5)
    box(d, 0, 140, 135, 64, "Part 5 · Feasibility & LLM",
        ["rules → Jev feasibility", "what-if router (Jev)", "tool-only chat + validator"], kind="panel",
        tsize=7.8, lsize=6.5)
    box(d, 180, 140, 302, 64, "Orchestrator · run_scenario(ScenarioSpec) → ScenarioResult",
        ["reruns Parts 1–4 for every what-if · Distributions + provenance", "cache · FastAPI · dashboard aggregation"],
        kind="green", tsize=7.8, lsize=6.6)
    for i, (t, ls) in enumerate([("3D twin viewer", ["TwinScene · retrofits appear"]),
                                 ("Customer app + chat", ["score · ladder · loan · memory"]),
                                 ("Lloyds dashboard", ["LSOA heat map · GNN · scenarios"])]):
        box(d, i * 168, 60, 146, 44, t, ls, kind="amber", tsize=7.8, lsize=6.5)
    # data arrows
    arrow(d, 67, 384, 67, 354)
    arrow(d, 247, 384, 247, 354)
    arrow(d, 416, 384, 416, 354)
    label(d, 71, 368, "Property, usage, weather", 6.0, GREY, "start")
    label(d, 251, 368, "NEED, catalogue", 6.0, GREY, "start")
    label(d, 420, 368, "live tables, sales", 6.0, GREY, "start")
    # core arrows
    arrow(d, 135, 300, 180, 300)
    label(d, 157, 304, "posterior", 5.8)
    label(d, 157, 290, "samples", 5.8)
    arrow(d, 315, 326, 350, 326)
    label(d, 332, 330, "energy,", 5.6)
    label(d, 332, 316, "rating", 5.6)
    arrow(d, 315, 264, 350, 264)
    label(d, 332, 268, "rating", 5.6)
    label(d, 332, 254, "after", 5.6)
    arrow(d, 120, 204, 195, 236)
    label(d, 150, 207, "FeasibilityReport", 6.0, GREY, "start")
    for x in (247, 416):
        arrow(d, x, 236, x, 204)
    arrow(d, 180, 165, 135, 165)
    label(d, 158, 169, "via tools", 5.8)
    arrow(d, 200, 140, 73, 104)
    arrow(d, 260, 140, 241, 104)
    arrow(d, 420, 140, 409, 104)
    # legend
    label(d, 0, 30, "Blue = coupled core (every energy number runs through the twin). Green = data + orchestration. "
                    "Grey = LLM side (reads only via tools).", 6.4, GREY, "start")
    label(d, 0, 18, "Parts 0, 3, 4, 5 and the UI are independent: each codes against its interface "
                    "(greentwin/interfaces.py) and a stub, then plugs in.", 6.4, GREY, "start")
    return d


# ----------------------------------------------------------------------------- diagram 2: end-to-end flow
def architecture():
    W, H = 482, 612
    d = Drawing(W, H)
    rh, gap = 46, 18
    full = W - 26
    ys = [H - 16 - rh - (rh + gap) * i for i in range(9)]
    n, g = 5, 6
    w = (full - (n - 1) * g) / n
    inputs = [("Property & energy", ["EPC records, NEED", "DESNZ LSOA, meters"]),
              ("EPC live tables", ["MHCLG D1/D3/EB4", "LA × quarter"]),
              ("Sales & market", ["UCL Price Paid–EPC", "UK HPI"]),
              ("Weather & geo", ["Open-Meteo, PVGIS", "ONSPD, IMD, planning"]),
              ("Customer input", ["EPC lookup, memory", "photos → Haiku vision"])]
    for i, (t, ls) in enumerate(inputs):
        box(d, i * (w + g), ys[0], w, rh, t, ls, tsize=7.4, lsize=6.1)
    label(d, -2, ys[0] + rh + 4, "INPUTS (PS §9: public + synthetic only)", 6.6, GREEN, "start", True)
    box(d, 0, ys[1], full, rh, "Part 0 · Data layer",
        ["download → map to contracts → property/customer splits (once) → leakage audit → DuckDB + Parquet + meta.json",
         "synthetic generator: real EPC homes → twin + behaviour noise → monthly usage + labels (validated vs NEED, D3)"],
        kind="green")
    w3 = (full - 16) / 3
    box(d, 0, ys[2], w3, rh, "Part 1 · Physics twin (JAX)", ["heat loss × degree days → kWh/fuel", "vmap over samples"], kind="core")
    box(d, w3 + 8, ys[2], w3, rh, "Part 1 · Bayesian calibration", ["NumPyro · home→LSOA→region", "posterior samples"], kind="core")
    box(d, 2 * (w3 + 8), ys[2], w3, rh, "Part 1 · 3D scene + capture", ["TwinScene, estimated heat loss", "Haiku vision → agent confirms"], kind="core")
    box(d, 0, ys[3], w3, rh, "Part 5 · Feasibility", ["rule vetoes → Jev", "FeasibilityReport"], kind="panel")
    box(d, w3 + 8, ys[3], full - w3 - 8, rh, "Part 2 · Energy & retrofit engine",
        ["EBM prior · baseline per fuel · ladder (marginal steps) · savings, carbon, bills",
         "improvement potential · rating after (SAP estimator) · Pareto + P(on front)"], kind="core")
    w2 = (full - 8) / 2
    box(d, 0, ys[4], w2, rh, "Part 3 · Green Home Score", ["score Distribution · P(band) · waterfall", "key issues · peer context (D1)"])
    box(d, w2 + 8, ys[4], w2, rh, "Part 4 · Property value", ["Bayesian hierarchical hedonic", "before vs after · indicative range"])
    box(d, 0, ys[5], full, rh, "Orchestrator · run_scenario(ScenarioSpec) → ScenarioResult",
        ["reruns Parts 1–4 for every what-if · every number a Distribution + provenance",
         "cached by hash(spec + model versions) · FastAPI · dashboard aggregation"], kind="green")
    box(d, 0, ys[6], w2, rh, "Part 5 · Knowledge base", ["SQL (numbers) · vector (docs)", "graph (rules, grants)"], kind="panel")
    box(d, w2 + 8, ys[6], w2, rh, "Part 5 · LLM chat + Jev router", ["tool-only · numeric validator", "template fallback"], kind="panel")
    for i, (t, ls) in enumerate([("3D twin viewer", ["react-three-fiber"]), ("Customer app + chat", ["ladder · loan · memory"]),
                                 ("Lloyds dashboard", ["LSOA heat map · GNN"])]):
        box(d, i * (w3 + 8), ys[7], w3, rh, t, ls, kind="amber")
    label(d, 0, ys[8] + rh - 8, "Every arrow is a contract in contracts/models.py; the experience layer never computes, "
                                "it renders ScenarioResult / TwinScene.", 6.5, GREY, "start")
    c3 = [w3 / 2, w3 + 8 + w3 / 2, 2 * (w3 + 8) + w3 / 2]
    mid = full / 2
    arrow(d, mid, ys[0], mid, ys[1] + rh)
    for c in c3:
        arrow(d, c, ys[1], c, ys[2] + rh)
    arrow(d, w3 + 8 + w3 / 2, ys[2], w3 + 8 + w3 / 2, ys[3] + rh)
    arrow(d, w3, ys[3] + rh / 2, w3 + 8, ys[3] + rh / 2)
    for c in (w2 / 2, w2 + 8 + w2 / 2):
        arrow(d, c, ys[3], c, ys[4] + rh)
        arrow(d, c, ys[4], c, ys[5] + rh)
        arrow(d, c, ys[5], c, ys[6] + rh)
    for c in c3:
        arrow(d, c, ys[6], c, ys[7] + rh)
    gx = W - 10
    d.add(Line(full, ys[7] + rh / 2, gx, ys[7] + rh / 2, strokeColor=BLUE, strokeWidth=1.1))
    d.add(Line(gx, ys[7] + rh / 2, gx, ys[5] + rh / 2, strokeColor=BLUE, strokeWidth=1.1))
    arrow(d, gx, ys[5] + rh / 2, full, ys[5] + rh / 2, color=BLUE, width=1.1)
    for k, t in enumerate(["what-if", "reruns", "models"]):
        label(d, gx + 1, ys[6] + rh / 2 + 8 - k * 8, t, 6.0, BLUE, "middle", True)
    return d


# ----------------------------------------------------------------------------- diagram 3: what-if flow
def whatif_flow():
    W, H = 482, 92
    d = Drawing(W, H)
    steps = [("User", ["3D toggle or", "chat question"]), ("Router", ["Jev / OpenAI", "→ ScenarioSpec"]),
             ("Feasibility", ["rules veto", "→ Jev"]), ("run_scenario", ["twin on every", "posterior sample"]),
             ("ScenarioResult", ["Distributions", "+ provenance"]), ("LLM + validator", ["quotes tool", "numbers only"]),
             ("UI", ["numbers, 3D,", "explanation"])]
    n, g = len(steps), 8
    w = (W - (n - 1) * g) / n
    for i, (t, ls) in enumerate(steps):
        kind = "core" if t == "run_scenario" else ("green" if t == "ScenarioResult" else "plain")
        box(d, i * (w + g), 20, w, 56, t, ls, kind=kind, tsize=7.4, lsize=6.4)
        if i < n - 1:
            arrow(d, i * (w + g) + w, 48, (i + 1) * (w + g), 48)
    label(d, 0, 6, "Low router confidence → clarifying question. Validator failure → regenerate once → template "
                   "explainer. Missing data → \"unavailable\" + reason.", 6.6, GREY, "start")
    return d


# ----------------------------------------------------------------------------- content
def part_block(title, tag, purpose, inside, models_, contracts_in, contracts_out, done):
    block = [Paragraph(title, S["h2"]), Paragraph("<font color='#5b5b5b'>%s</font>" % tag, S["small"]),
             Paragraph("<b>Purpose.</b> " + purpose, S["body"])]
    block += [Paragraph(s, S["bullet"], bulletText="•") for s in inside]
    rows = [["Models (primary · baseline)", models_], ["Consumes", contracts_in], ["Produces", contracts_out],
            ["Done when", done]]
    data = [[Paragraph("<b>%s</b>" % a, S["cell"]), Paragraph(b, S["cell"])] for a, b in rows]
    t = Table(data, colWidths=[38 * mm, 132 * mm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, GRID), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("BACKGROUND", (0, 0), (0, -1), GREEN_L), ("TOPPADDING", (0, 0), (-1, -1), 3),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    block += [t, Spacer(1, 8)]
    story.append(CondPageBreak(70 * mm))
    story.append(KeepTogether(block))


def build():
    from datasets import DATASETS, DAY1, EPC_LIVE_USES  # noqa: E402

    # ---------------------------------------------------------------- cover
    story.append(Spacer(1, 34 * mm))
    P("GreenTwin", "title")
    P("System design: five parts, their contracts, data and build plan", "subtitle")
    story.append(Spacer(1, 4 * mm))
    P("Megathon X: Singularity · Lloyds Problem Statement 2: Green Home Score and Retrofit Recommendation "
      "Engine · Team MegaMind · contracts schema 0.4.0", "meta")
    story.append(Spacer(1, 8 * mm))
    callout("<b>In one line.</b> A Bayesian-calibrated physics twin of every home, which you can see in 3D, "
            "explain in plain language and act on, where every number is backed by physics and the range it "
            "comes with.")
    story.append(Spacer(1, 4 * mm))
    P("<b>Audience:</b> the build team. This document is the high-level design: the five parts, how they connect, "
      "the data that feeds them and the build plan. The companion <b>LLD.pdf</b> holds every contract field, "
      "interface signature, algorithm, table schema and download step. Binding rules: <font face='%s'>AGENTS.md</font> "
      "and <font face='%s'>contracts/</font>." % (F, F))
    story.append(Spacer(1, 4 * mm))
    table([
        ["Section", "What it covers"],
        ["1. Philosophies", "The six principles every part follows; how the brief's core rule is enforced"],
        ["2. The five parts", "What each part answers, what is inside it, coupled core vs independent parts"],
        ["3. Architecture", "End-to-end flow, the what-if loop, technology choices"],
        ["4. Part designs", "Purpose, components, models, contracts in/out and done-when for each part"],
        ["5. Contracts and interfaces", "The contract and Protocol each part owns (full detail in LLD.pdf)"],
        ["6. Data", "Datasets and how to get them, the EPC live tables, joins, synthetic data, day-1 plan"],
        ["7. Uncertainty", "Every number as a full distribution"],
        ["8. Key mechanisms", "Improvement ladder, improvement potential"],
        ["9. Evaluation and submission", "Metrics vs baselines, PS coverage, submission items"],
        ["10. Build plan", "Work breakdown, checkpoints, coding guidelines, open decisions"],
    ], [48 * mm, 122 * mm], zebra=False)
    story.append(PageBreak())

    # ---------------------------------------------------------------- 1 philosophies
    H1("1. Core philosophies")
    table([
        ["#", "Philosophy", "What it means in the system"],
        ["1", "<b>Physics is the source of truth</b>", "Every energy, savings, carbon and rating number comes from the "
         "Level-1 digital twin (Part 1). ML sets priors (EBM), estimates value and checks results; it never produces "
         "a customer's savings, so the system handles retrofit combinations no dataset has seen."],
        ["2", "<b>Bayesian in thought</b>", "Each home is calibrated on its own meter readings. Every output is a full "
         "distribution (mean, median, mode, std, percentiles, 90% interval, tail probabilities) plus P(Poor), "
         "P(suitable), P(on front). Missing data never becomes zero; it widens the range."],
        ["3", "<b>Rerun, never add up</b>", "A retrofit is a change to the twin's parameters; every what-if and package "
         "reruns the model. Interactions come from the physics. Savings are always per fuel."],
        ["4", "<b>Grounded explanation</b>", "The LLM only quotes tool outputs; a validator rejects any other number; Jev "
         "only routes questions and screens feasibility. No backing → \"unavailable\"."],
        ["5", "<b>Explainability first</b>", "Score waterfall, heat loss per element, EBM contributions, value "
         "coefficients, a reason on every recommendation or veto."],
        ["6", "<b>See it to believe it</b>", "A living 3D twin: heat loss glows, retrofits appear, numbers update live, "
         "all driven by the same calibrated model."],
    ], [8 * mm, 40 * mm, 122 * mm])
    H2("The brief's core rule, enforced by design")
    P("<i>\"The score, savings, carbon and value figures come from documented models and calculations. The LLM only "
      "explains them... For a what-if request, the relevant models must first be rerun.\"</i> (PS §1)")
    bullets([
        "Only Parts 1–4 produce numbers, each with <font face='%s'>provenance</font> naming its model." % F,
        "Every what-if goes through <b>run_scenario</b> before the LLM sees anything.",
        "The LLM's answer passes a numeric validator → regenerate once → template explainer.",
    ])
    story.append(PageBreak())

    # ---------------------------------------------------------------- 2 the five parts
    H1("2. The five parts")
    P("GreenTwin is five parts on top of a shared data layer and a thin orchestrator. Parts 1 and 2 form the "
      "<b>coupled core</b>: every energy number runs through the twin, so they are built by one team in sequence. "
      "Parts 3, 4 and 5 (and the data layer and UI) are <b>independent</b>: each codes against its interface and a "
      "stub, and connects to the core at a few fixed points.")
    story.append(parts_overview())
    story.append(Spacer(1, 6))
    table([
        ["Part", "Answers (PS)", "Inside", "Coupling"],
        ["<b>1 · Digital twin</b>", "the physics behind §3, §5, §6", "JAX physics twin · NumPyro Bayesian calibration · "
         "3D scene · photo-to-twin capture · electricity layer", "Core"],
        ["<b>2 · Energy & retrofit</b>", "§3 potential, §4 retrofits, §5 savings, §6 carbon", "EBM energy prior · baseline · "
         "improvement ladder · savings, carbon, bills · improvement potential · rating after · Pareto", "Core"],
        ["<b>3 · Green Home Score</b>", "§2 score, §3 poor homes", "fixed-weight score · sub-scores · waterfall · key "
         "issues · projected score · peer context", "Independent"],
        ["<b>4 · Property value</b>", "§7 value", "Bayesian hierarchical hedonic model · before vs after · green twins",
         "Independent"],
        ["<b>5 · Feasibility & LLM</b>", "§1 core rule, §4 feasibility, §8 what-if, §10 memory", "rules → Jev feasibility "
         "· what-if router · tool-only chat + validator · knowledge base · memory", "Independent"],
    ], [32 * mm, 36 * mm, 78 * mm, 24 * mm])
    H2("The four connection points")
    table([
        ["From → to", "Contract", "Until it exists, use"],
        ["Part 2 EBM → Part 1 calibration", "EnergyPrior (prior mean per fuel)", "RdSAP defaults"],
        ["Part 5 feasibility → Part 2 ladder", "FeasibilityReport", "rules only"],
        ["Part 2 rating → Parts 3, 4", "RatingEstimate (estimated SAP after)", "stub: band + 1"],
        ["Orchestrator → Part 5 LLM, UI", "ScenarioResult, TwinScene", "contracts/examples/*_MOCK.json"],
    ], [52 * mm, 70 * mm, 48 * mm])
    story.append(PageBreak())

    # ---------------------------------------------------------------- 3 architecture
    H1("3. Architecture (end to end)")
    story.append(architecture())
    story.append(PageBreak())
    H2("The what-if loop")
    story.append(whatif_flow())
    story.append(Spacer(1, 6))
    H2("Technology choices")
    table([
        ["Concern", "Choice"],
        ["Data + storage", "Python 3.12, pandas/Polars, DuckDB + Parquet with a meta.json per file"],
        ["Physics twin", "<b>JAX</b> (jax.numpy): runs inside the NumPyro model so NUTS/SVI get exact gradients; "
         "jax.vmap over posterior samples. Monthly steady-state RC model; differentiable RC graph (diffrax) later"],
        ["Bayesian", "NumPyro: NUTS offline for the hierarchy, SVI per home (&lt; 1 s); ArviZ diagnostics"],
        ["Electricity layer", "Half-hourly: smart-meter load-shape archetypes + heat pump + PVGIS PV − battery"],
        ["Learned models", "EBM (InterpretML) energy prior, XGBoost baseline; GraphSAGE GNN (PyTorch Geometric) on the "
         "LSOA graph; Bayesian hedonic value model (NumPyro)"],
        ["Photo-to-twin", "Claude Haiku 5.5 (claude-haiku-5-5) vision + structured outputs; react-konva editor"],
        ["LLM + router", "OpenAI Responses API tool use (LLM_MODEL = gpt-6-luna) + numeric validator + template "
         "fallback; Jev System 1 if a key exists, else OpenAI structured output"],
        ["API + front end", "FastAPI; React + Vite + TypeScript, react-three-fiber + drei, Recharts, MapLibre + "
         "deck.gl, Zustand; types generated from the contracts"],
    ], [32 * mm, 138 * mm])
    story.append(PageBreak())

    # ---------------------------------------------------------------- 4 part designs
    H1("4. Part designs")
    part_block(
        "Part 0 · Data layer", "PS §9 · feeds every part",
        "Clean, licensed, split, joinable real data and every external factor traceable to a cited source.",
        ["Download scripts write raw files untouched, then Parquet + meta.json (DatasetMeta) per source.",
         "EPC record → Property mapper: regex rules turn EPC text (walls, roof, glazing, heating, controls) into "
         "contract enums; ≥ 95% mapped per field; unmapped values logged.",
         "EPC live tables (MHCLG D1/D3/EB4) → AreaEPCStats per local authority × quarter: calibration targets, "
         "validation checks, peer context and dashboard layers (section 6).",
         "Splits: 70/15/15 by customer, else property, seed 42, assigned once; spatial-block splits for area models; "
         "leakage audit (epc.potential_* are labels only).",
         "Synthetic generator (uses the Part 1 twin): real EPC homes + NEED-calibrated behaviour → monthly usage with "
         "gaps and conflicts + reference labels; reweighted to EPC live-table band and type mixes; validated vs NEED."],
        "— (data engineering); EPC mapper, k-means load profiles",
        "public datasets, customer input", "Property, EnergyUsageMonth, SaleRecord, WeatherMonth, AreaEPCStats, "
        "LSOAConsumption, LoadProfileArchetype, EPCRecommendation, ReferenceLabel, DatasetMeta",
        "No id spans two splits; mapper coverage ≥ 95%; our EPC counts reconcile with live table D1; every table has "
        "meta.json; DATA section of the LLD complete.")
    part_block(
        "Part 1 · Digital twin (physics, Bayesian, 3D)", "PS §3, §5, §6 physics · coupled core",
        "A documented physics model of each home, calibrated on its own bills, that every energy number runs through, "
        "and that the customer can see in 3D.",
        ["Physics (JAX): heat loss Σ U·A + ventilation × degree days → space heat; hot water; baseload; fuel via "
         "boiler efficiency or heat-pump SCOP; PV cuts imports. Calibrated and standard (SAP occupancy) modes.",
         "Bayesian calibration (NumPyro): heat-loss multiplier, setpoint, baseload, efficiency; hierarchy home → LSOA → "
         "region; Student-t likelihood on monthly meters; missing months latent; SVI per home, NUTS offline.",
         "3D scene: parametric house (or captured rooms) with surfaces coloured by estimated heat-loss share and "
         "retrofit slots; never calculates.",
         "Photo-to-twin: Haiku vision drafts rooms + element tags with confidence; agent confirms in a react-konva "
         "editor; room geometry improves areas and the 3D house.",
         "Electricity layer: half-hourly profile for PV self-consumption, battery and time-of-use bills."],
        "Level-1 RC twin (physics) · Bayesian calibration (vs degree-day OLS, LightGBM quantile)",
        "Property, EnergyUsageMonth, WeatherMonth, EnergyPrior, LoadProfileArchetype",
        "TwinParams, CalibrationResult, PosteriorSamples (in-process), ElementHeatLoss, TwinScene, CaptureResult, "
        "ElectricityProfileSummary",
        "SH1–SH3 within ±15% of metered totals; monotone sweep passes; 90% coverage ≈ 90% on held-out homes; "
        "standard-mode kWh/m² tracks EPC live table D3 by local authority.")
    part_block(
        "Part 2 · Energy calculation & retrofit efficiency", "PS §3, §4, §5, §6 · coupled core",
        "Turn the calibrated twin into the numbers the brief asks for: baseline energy per fuel, what each retrofit "
        "adds, savings, carbon, bills, improvement potential, the estimated rating after, and the best packages.",
        ["Energy prior: EBM on NEED predicts kWh/m² per fuel (prior for homes without meters) with feature "
         "contributions.",
         "Baseline: per-fuel Distributions; intensity = annualised metered energy per m² (the brief's definition).",
         "Improvement ladder: feasible retrofits in fabric-first order; each step's marginal effect on every posterior "
         "sample; P(suitable) = P(feasible) × P(net benefit); probabilistic verdicts.",
         "Improvement potential: all feasible retrofits in one run; null + reason if the baseline is 0.",
         "Rating after: SAP 2012 formula on the twin's standard mode, anchored to the recorded SAP.",
         "Pareto: enumerate feasible packages; P(on front) across samples; knee."],
        "EBM energy prior (vs NEED median, XGBoost) · SAP estimator (vs median band shift) · Pareto (vs greedy kWh/£)",
        "PosteriorSamples, FeasibilityReport, catalogue, emission factors, tariffs",
        "EnergyPrior, EnergyBreakdown, RetrofitEffect, ImprovementPotential, RatingEstimate, ParetoPoint",
        "SH2 shows electricity rising after a heat pump; package = one run; potential null at baseline 0; rating "
        "change validated on EPC before/after pairs; precision@k vs EPC assessor recommendations.")
    part_block(
        "Part 3 · Green Home Score", "PS §2, §3 · independent",
        "Answer \"how green is my home\" with the brief's fixed weights, a transparent waterfall and the issues that "
        "matter.",
        ["S = Σ w·s from contracts/score_mapping.json; seven monotonic, documented sub-scores; rounded; five bands.",
         "Energy efficiency from the recorded SAP (or the estimated rating for a projected score); intensity from "
         "annualised metered kWh/m².",
         "Score Distribution and P(each band) from posterior samples; key issues with evidence; peer context from "
         "the EPC live tables (local-authority band shares, D1)."],
        "Deterministic scoring (no training)", "Property, EnergyBreakdown, RatingEstimate, AreaEPCStats",
        "ScoreResult (components, KeyIssue list, band probabilities)",
        "Weights sum to 1; monotonicity sweep; every heating type mapped; projected ≥ current for improvement-only "
        "packages.")
    part_block(
        "Part 4 · Property value", "PS §7 · independent",
        "An indicative value range, before vs after, accounting for location, size, type and market.",
        ["Bayesian hierarchical hedonic regression on real UK sales: log(price ÷ HPI) = pooled location effects "
         "(LSOA within LA within region) + size + type + age + tenure + β(region × type) × SAP.",
         "Same home at its recorded vs estimated-after rating, applied to the current value estimate; full "
         "distribution; wording always \"indicative\"; range including 0 → no clear effect.",
         "Green twins: matched greener comparable homes shown as ranges."],
        "Bayesian hedonic (vs OLS without the rating)", "Property, SaleRecord (training), RatingEstimate",
        "ValueImpact, GreenTwin", "Held-out coverage and pinball loss beat the baseline; β compared with Nationwide / "
        "DECC 2013; planted-premium recovery reported as a method check only.")
    part_block(
        "Part 5 · Feasibility rules engine, Jev, LLM explanation and chat", "PS §1, §4, §8, §10 · independent",
        "Decide what is possible for a home, route what-if questions, and explain computed numbers without ever "
        "inventing one.",
        ["Feasibility: rule vetoes from contracts/feasibility_rules.json always win; Jev classifies survivors with "
         "recalibrated confidence; low confidence → needs survey; OpenAI structured output without a Jev key.",
         "Router: Jev intent + ScenarioSpec slots; below the threshold → clarifying question.",
         "Chat: tool-only gpt-6-luna (get_home_facts, get_energy_prior, run_scenario, rank_recommendations, search_docs, query_graph, memory tools), numeric "
         "validator, regenerate once, template fallback; knowledge base (SQL, vector, graph); customer memory."],
        "Jev threshold calibration (isotonic) · LLM evaluated on ≥ 30 adversarial scenarios",
        "Property, catalogue, CustomerMemory, ScenarioResult (via tools)",
        "FeasibilityReport, WhatIfRoute, ChatResponse, ExplanationCheck, CustomerMemory",
        "Rule veto beats Jev; 0 unvalidated numbers; groundedness and accuracy ≥ 90%, unsupported claims &lt; 5%; "
        "works with no keys.")
    story.append(PageBreak())

    # ---------------------------------------------------------------- 5 contracts and interfaces
    H1("5. Contracts and interfaces")
    P("Parts exchange data only through Pydantic contracts (<font face='%s'>contracts/models.py</font>, schema 0.4.0; "
      "TypeScript mirror generated into <font face='%s'>contracts/types.ts</font>) and call each other only through "
      "the Protocols in <font face='%s'>greentwin/interfaces.py</font>. Every field and signature is listed in "
      "LLD.pdf, generated from the code." % (F, F, F))
    table([
        ["Part", "Interface (Protocol)", "Key methods", "Owns these contracts"],
        ["0 Data", "DataLayer", "load_property, load_usage, load_weather, load_sales, load_area_epc_stats, "
         "load_lsoa_consumption, load_load_profiles, catalogue", "Property, EnergyUsageMonth, SaleRecord, WeatherMonth, "
         "AreaEPCStats, LSOAConsumption, LoadProfileArchetype, EPCRecommendation, ReferenceLabel, DatasetMeta"],
        ["1 Twin", "DigitalTwin", "params_from_property, apply_retrofits, simulate, calibrate, element_heat_loss, "
         "electricity_profile, scene, capture", "TwinParams, CalibrationResult, ElementHeatLoss, TwinScene, "
         "SceneSurface, RetrofitSlot, FloorPlan, Room, ElementTag, CaptureResult"],
        ["2 Energy", "EnergyRetrofitEngine", "energy_prior, baseline, ladder, package, improvement_potential, "
         "estimate_rating, pareto", "EnergyPrior, EnergyBreakdown, MonthlyEnergy, ElectricityProfileSummary, "
         "RetrofitEffect, ImprovementPotential, RatingEstimate, ParetoPoint"],
        ["3 Score", "GreenHomeScorer", "score", "ScoreResult, ScoreComponent, KeyIssue"],
        ["4 Value", "ValueModel", "value_impact, green_twins", "ValueImpact, GreenTwin"],
        ["5 Feasibility & LLM", "FeasibilityEngine, Router, Explainer", "assess, route, chat", "Feasibility, "
         "FeasibilityReport, WhatIfRoute, ChatRequest, ChatResponse, ExplanationCheck, Citation, CustomerMemory"],
        ["Orchestrator", "Orchestrator", "run_scenario, dashboard_areas", "ScenarioSpec, ScenarioResult, "
         "DashboardArea"],
        ["Shared", "contracts.summarise", "summarise, exact, category_probabilities", "Distribution, Provenance, "
         "FeatureContribution"],
    ], [24 * mm, 32 * mm, 56 * mm, 58 * mm])
    callout("<b>Rule for every part:</b> uncertain numbers are Distributions built with contracts.summarise; every "
            "result carries Provenance; missing data is None (never 0); savings are per fuel; packages are re-run, "
            "never summed. A test fails if types.ts drifts from models.py.")
    story.append(PageBreak())

    # ---------------------------------------------------------------- 6 data
    H1("6. Data")
    P("No public dataset links property features, monthly meter readings and sale prices for the same homes. So: "
      "<b>real data for every part that exists publicly</b>, joined at the level each dataset allows, plus "
      "<b>physics-generated synthetic data only for the missing piece</b> (monthly readings per home and the "
      "labels), calibrated and validated against real aggregates.")
    H2("Datasets and how to get them")
    table([["Dataset", "Used by", "Access", "Licence"]] +
          [[f"<b>{x['name']}</b>", x["use"], x["access"], x["licence"]] for x in DATASETS],
          [52 * mm, 38 * mm, 54 * mm, 26 * mm])
    story.append(CondPageBreak(55 * mm))
    H2("Start here (day 1, in this order)")
    for i, s in enumerate(DAY1, 1):
        story.append(Paragraph(f"{i}. {s}", S["bullet"]))
    story.append(Spacer(1, 4))
    story.append(CondPageBreak(55 * mm))
    H2("The EPC live tables (MHCLG): how they are used")
    P("The GOV.UK <i>Live tables on Energy Performance of Buildings Certificates</i> are quarterly ODS tables of EPC "
      "<b>lodgements</b> for England and Wales by local authority × quarter (D1: rating band counts; D3: floor area, "
      "modelled energy kWh/m², CO2, fuel cost; EB4: property type; EB-tables for existing dwellings only). They are "
      "aggregates, not homes, so they are used to anchor and check the per-home data, not to train per-home models:")
    table([["Use", "Table", "How"]] + [[u, t, h] for u, t, h in EPC_LIVE_USES], [44 * mm, 18 * mm, 108 * mm])
    P("Caveats handled: counts are lodgements in a period (repeat certificates, opt-outs), not the dwelling stock; "
      "a Q1 2026 local-authority allocation error was revised in May 2026; D3 averages were corrected in December "
      "2023 (recompute averages from totals as the table notes explain).", "small")
    story.append(CondPageBreak(55 * mm))
    H2("How the data joins")
    table([
        ["Level", "Join", "Datasets"],
        ["Postcode / LSOA", "ONS postcode directory → LSOA 2021, LA code, region", "EPC records, UCL sales, weather "
         "(by district), planning, off-gas, IMD, DESNZ LSOA"],
        ["Local authority × quarter", "LA code", "EPC live tables ↔ our EPC sample, synthetic population, twin "
         "standard-mode output, dashboard"],
        ["Archetype", "type × age band × floor-area band × region", "NEED (no postcode) ↔ EPC homes, EBM training, "
         "synthetic validation"],
        ["Per home (synthetic)", "real EPC home + real weather → twin + behaviour noise", "monthly usage + labels"],
    ], [34 * mm, 60 * mm, 76 * mm])
    story.append(PageBreak())

    # ---------------------------------------------------------------- 7 uncertainty
    H1("7. Uncertainty: every number is a distribution")
    P("Bayesian inference happens in two places: calibrating each home's physics on its bills (Part 1) and the value "
      "model on real sales (Part 4). Everything downstream <b>propagates</b> those posteriors: each output is "
      "computed once per posterior sample, then summarised with contracts.summarise. Intervals are never added "
      "together; the samples carry the correlations.")
    table([
        ["Part", "Fields", "Why"],
        ["Centre", "mean · median · mode", "They differ when the result is lopsided"],
        ["Spread", "standard deviation · IQR", "How uncertain"],
        ["Percentiles + intervals", "P1–P99 · 50% and 90% credible intervals", "Full shape; the 90% interval is the headline"],
        ["Tails", "labelled tail probabilities · mean of the worst 5% · skewness", "What could go wrong, and how badly"],
        ["Shape + meta", "histogram · unit · n_samples · headline (mean or median)", "Density plot; traceability"],
    ], [34 * mm, 74 * mm, 62 * mm])
    table([
        ["Mean", "Median", "Mode", "Std", "P5", "P95", "P(saving &gt; 0)", "P(rise &gt; £100)"],
        ["−£35", "−£42", "−£55", "£88", "−£180", "+£110", "0.38", "0.23"],
    ], [18 * mm, 18 * mm, 18 * mm, 18 * mm, 20 * mm, 20 * mm, 29 * mm, 29 * mm], zebra=False)
    P("Example (illustrative): heat-pump bill change on SH2. The mean hides that about 1 in 4 homes like this could "
      "see the bill rise by over £100/yr while carbon falls with near certainty.", "small")

    # ---------------------------------------------------------------- 8 mechanisms
    H1("8. Key mechanisms")
    story.append(CondPageBreak(55 * mm))
    H2("Improvement ladder: what each retrofit adds")
    P("Rules + Jev decide what is possible; the twin decides what it is worth. Step k's effect = twin(steps 1..k) − "
      "twin(steps 1..k−1) on every sample, fabric first. The steps add up exactly to the package (one run). Verdicts "
      "are probabilistic: Recommended if P(suitable) ≥ 0.8 and P(step adds ≥ 3% of purchased energy or ≥ 2 points) "
      "≥ 0.8. Thresholds to be agreed.")
    table([
        ["#", "Add (SH2, illustrative)", "Gas kWh/yr", "Elec kWh/yr", "Potential", "Score", "CO2 t", "Bill £", "Verdict"],
        ["1", "Cavity wall insulation", "2,600", "0", "17%", "66", "0.48", "+207", "Recommended"],
        ["2", "Loft top-up to 270 mm", "900", "0", "23%", "68", "0.16", "+72", "Recommended"],
        ["3", "Draught proofing", "400", "0", "26%", "69", "0.07", "+32", "Optional"],
        ["4", "Smart thermostat + TRVs", "500", "0", "29%", "71", "0.09", "+40", "Recommended"],
        ["5", "Replace old double glazing", "450", "0", "32%", "72", "0.08", "+36", "Not worth adding"],
        ["6", "Air-source heat pump", "7,150", "−2,300", "64%", "80", "0.86", "−35", "Depends on goal"],
        ["7", "Solar PV 3 kWp", "0", "1,100", "71%", "86", "0.22", "+290", "Recommended"],
        ["", "<b>Package (one run)</b>", "<b>12,000</b>", "<b>−1,200</b>", "<b>71%</b>", "<b>86</b>", "<b>1.96</b>",
         "<b>+641</b>", ""],
    ], [7 * mm, 38 * mm, 18 * mm, 18 * mm, 16 * mm, 12 * mm, 12 * mm, 14 * mm, 35 * mm])
    story.append(CondPageBreak(55 * mm))
    H2("Improvement potential (PS §3)")
    bullets(["100 × (E_baseline − E_after) / E_baseline on purchased energy (grid import + gas), E_after from all "
             "feasible retrofits run together; a distribution with the per-fuel split.",
             "Needs-survey items reported separately; baseline 0 or missing → unavailable + reason; homes ranked by "
             "posterior mean with P(top 10%)."])
    story.append(PageBreak())

    # ---------------------------------------------------------------- 9 evaluation
    H1("9. Evaluation and submission")
    table([
        ["Output", "Metric (held-out, property splits)", "Baseline", "Part"],
        ["Energy prior", "MAE/MAPE per fuel, 90% coverage", "NEED median, XGBoost", "2"],
        ["Calibration", "coverage ≈ 90%, CRPS, R-hat; standard mode vs EPC D3 by LA", "degree-day OLS", "1"],
        ["Rating after", "MAE of SAP change on EPC before/after pairs", "median band shift", "2"],
        ["Recommendations", "precision@k vs EPC assessor lists; 0 infeasible", "recommend all feasible", "2, 5"],
        ["Improvement ranking", "Spearman / NDCG vs labels", "rank by total kWh", "2"],
        ["Score", "0 monotonicity violations; band-probability calibration", "—", "3"],
        ["Value", "coverage, pinball loss; β vs published", "OLS without rating", "4"],
        ["GNN area model", "MAE on held-out LSOAs (spatial blocks)", "EBM without graph", "0/dashboard"],
        ["Photo capture", "room-area error, tag accuracy, % fields corrected", "manual tagging", "1"],
        ["LLM", "groundedness ≥ 90%, accuracy ≥ 90%, unsupported &lt; 5%", "no validator", "5"],
        ["Synthetic data", "KS vs NEED; band/type mix vs EPC live tables", "—", "0"],
    ], [30 * mm, 70 * mm, 40 * mm, 30 * mm])
    table([
        ["PS", "Requirement", "Part"],
        ["§1", "Core rule; what-if reruns models first", "5 + orchestrator"],
        ["§2", "Score with fixed weights, documented monotonic sub-scores, contributions", "3"],
        ["§3", "Poor homes, key issues, improvement potential per fuel", "3 + 2"],
        ["§4", "Feasible, prioritised retrofits with suitability, saving, reason", "5 + 2"],
        ["§5 · §6", "Per-fuel savings with interactions; carbon from documented factors", "1 + 2"],
        ["§7", "Indicative value range, before vs after", "4 (+ 2 rating)"],
        ["§9", "Data, splits, leakage, documented synthetic labels", "0"],
        ["§10", "Customer memory changes recommendations", "5"],
        ["§13", "A prototype · B code · C models · D score doc · E LLM doc · F this document · G eval · H RAI · I "
         "test predictions", "all"],
    ], [18 * mm, 122 * mm, 30 * mm])
    story.append(PageBreak())

    # ---------------------------------------------------------------- 10 build plan
    H1("10. Build plan")
    H2("Work breakdown")
    table([
        ["Track", "Sequence", "Starts on"],
        ["<b>Coupled core</b> (Parts 1 + 2, one team)", "physics twin (JAX) → rating estimator → synthetic generator → "
         "Bayesian calibration → ladder, potential, Pareto → electricity layer", "RdSAP defaults, SH1–SH3 examples"],
        ["Part 0 Data", "EPC access + downloads → mapper → splits → DuckDB → live tables + DESNZ + smart meters", "day 1"],
        ["Part 3 Score", "score + monotonic tests → key issues → peer context → projected score", "recorded inputs"],
        ["Part 4 Value", "UCL sales prep → Bayesian hedonic → before/after → green twins", "UCL sample; stub rating"],
        ["Part 5 Feasibility & LLM", "rules engine → mock server → chat + validator → eval set → Jev → memory", "mocks"],
        ["Experience", "app shell on mocks → 3D twin v1 → capture editor → dashboard", "mocks, generated types"],
    ], [44 * mm, 90 * mm, 36 * mm])
    table([
        ["Checkpoint", "Done when"],
        ["1. End-to-end skeleton", "SH1–SH3 flow through the real twin → score → retrofits → API → UI"],
        ["2. Complete submission", "Bayesian distributions, rating estimator, value, grounded LLM, evaluation, docs"],
        ["3. Demo build (freeze)", "3D twin + capture, Pareto + loan, Lloyds dashboard, memory"],
    ], [42 * mm, 128 * mm])
    story.append(CondPageBreak(55 * mm))
    H2("Working rules (hackathon: speed first)")
    bullets([
        "Speed beats polish: thinnest end-to-end path first, mocks from day one, reuse and hardcode for the demo, "
        "timebox at ~45 minutes, commit small and often.",
        "Tests only where judges look: contracts, monotonicity, per-fuel signs, null potential at baseline 0, the "
        "LLM validator, no mock in the demo, types.ts in sync.",
        "Non-negotiables: contracts in and out; provenance; Distributions via summarise; missing ≠ 0; rerun never "
        "sum; the LLM never makes a number; value always indicative; no keys in git.",
        "One UI look: shared theme and components (NumberWithRange, DistributionCard, ScoreWaterfall), brand green "
        "#0b6e4f / blue #2a6fdb, colour-blind-safe scales, every number with unit + range.",
    ])
    story.append(CondPageBreak(55 * mm))
    H2("Open decisions")
    table([
        ["Decision", "Options", "Recommendation"],
        ["Twin ambition (library: JAX)", "A: monthly steady-state first · B: differentiable RC graph (diffrax) first",
         "A for checkpoint 1; upgrade behind simulate()"],
        ["Verdict thresholds", "gain threshold, probability cut-offs, cost per kWh", "Agree once; record in SCORE doc"],
        ["Default goal", "bills or carbon", "Bills by default; carbon toggle"],
        ["Time-of-use tariff source", "which published tariff to cite", "Pick one public tariff; until then null"],
    ], [40 * mm, 76 * mm, 54 * mm])


def footer(c, doc):
    c.saveState()
    c.setFont(F, 7.5)
    c.setFillColor(GREY)
    c.drawString(20 * mm, 10 * mm, "GreenTwin · Team MegaMind · Megathon X, Lloyds PS2 · System design")
    c.drawRightString(A4[0] - 20 * mm, 10 * mm, f"{doc.page}")
    c.restoreState()


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    build()
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=16 * mm, bottomMargin=18 * mm,
                            title="GreenTwin — System design", author="Team MegaMind")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
