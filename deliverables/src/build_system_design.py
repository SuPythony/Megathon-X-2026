"""Build deliverables/SYSTEM_DESIGN.pdf: GreenTwin system design and decoupled build blocks.

Run:  python deliverables/src/build_system_design.py
Needs reportlab. Uses Segoe UI (Windows) or DejaVu Sans (Linux), else Helvetica.
"""

import math
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
OUT = ROOT / "deliverables" / "SYSTEM_DESIGN.pdf"


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


# ----------------------------------------------------------------------------- diagram 1: architecture
def architecture():
    W, H = 482, 616
    d = Drawing(W, H)
    rh, gap = 52, 20
    xl = 0
    full = W - 26  # leave a right gutter for the what-if loop
    ys = [H - 16 - rh - (rh + gap) * i for i in range(8)]  # bottom of each row, top -> bottom

    # Row 1: inputs
    n, g = 5, 6
    w = (full - (n - 1) * g) / n
    inputs = [
        ("Property & energy", ["EPC + recommendations", "NEED metered energy"]),
        ("Sales & market", ["UCL Price Paid–EPC", "UK House Price Index"]),
        ("Weather & solar", ["HadUK-Grid, Open-Meteo", "PVGIS"]),
        ("Geo + factors", ["ONS LSOA, IMD, planning", "DESNZ, Ofgem, BEIS"]),
        ("Customer input", ["EPC lookup, form, agent", "photos + memory"]),
    ]
    for i, (t, ls) in enumerate(inputs):
        box(d, xl + i * (w + g), ys[0], w, rh, t, ls, tsize=7.6, lsize=6.3)
    label(d, -2, ys[0] + rh + 4, "INPUTS (PS §9: public + synthetic only)", 6.6, GREEN, "start", True)

    # Row 2: data foundation
    box(d, xl, ys[1], full, rh, "B1  Data foundation",
        ["map to contracts · property/customer splits (70/15/15, once) · leakage audit (epc.potential_*)",
         "annualise metered usage (missing months flagged, never 0) · sourced factors · DuckDB + Parquet"],
        kind="green")

    # Row 3: twin trio
    w3 = (full - 2 * 8) / 3
    box(d, xl, ys[2], w3, rh, "B2  Synthetic generator",
        ["real EPC homes → twin + behaviour", "usage + reference labels, vs NEED"])
    box(d, xl + w3 + 8, ys[2], w3, rh, "B2  Level-1 physics twin",
        ["heat loss ΣU·A + air × degree days", "→ COP / efficiency → kWh per fuel"], kind="core")
    box(d, xl + 2 * (w3 + 8), ys[2], w3, rh, "B2  Rating estimator",
        ["SAP 2012 formula on twin (standard)", "anchored to recorded SAP"])

    # Row 4: Bayesian
    box(d, xl, ys[3], full, rh, "B3  Bayesian calibration + energy prediction",
        ["EBM prior (no meter data) → NumPyro hierarchical calibration on the home's meter readings",
         "→ posterior samples → electricity + gas kWh/yr as a full Distribution (mean, mode, std, P1–P99, tails)"],
        kind="core")

    # Row 5: score / retrofit / value
    box(d, xl, ys[4], w3, rh, "B4  Green Home Score",
        ["fixed weights, monotonic sub-scores", "waterfall · P(each band) · issues"])
    box(d, xl + w3 + 8, ys[4], w3, rh, "B4  Retrofit engine",
        ["rules → Jev feasibility → twin reruns", "ladder · P(suitable) · Pareto"])
    box(d, xl + 2 * (w3 + 8), ys[4], w3, rh, "B5  Property value",
        ["Bayesian hierarchical hedonic", "before vs after · indicative range"])

    # Row 6: scenario engine
    box(d, xl, ys[5], full, rh, "B6  Scenario engine: run_scenario(ScenarioSpec) → ScenarioResult",
        ["reruns B2–B5 for every what-if · per-fuel savings, carbon, improvement potential, bills, value",
         "every number = Distribution + provenance · cached by hash(spec + model versions) · FastAPI"],
        kind="green")

    # Row 7: KB + LLM
    wk = (full - 8) / 2
    box(d, xl, ys[6], wk, rh, "B6  Knowledge base",
        ["SQL (numbers) · vector (docs, grants)", "graph (rules, constraints, regions)"], kind="panel")
    box(d, xl + wk + 8, ys[6], wk, rh, "B6  Grounded LLM + Jev router",
        ["tool-only · numeric validator · template", "fallback · Jev routes + screens feasibility"],
        kind="panel")

    # Row 8: experience
    box(d, xl, ys[7], w3, rh, "B7  3D digital twin",
        ["heat loss glows (estimated)", "retrofits appear, numbers update"], kind="amber")
    box(d, xl + w3 + 8, ys[7], w3, rh, "B7  Customer app + chat",
        ["score · ladder · Pareto · loan", "memory · DistributionCard"], kind="amber")
    box(d, xl + 2 * (w3 + 8), ys[7], w3, rh, "B8  Lloyds dashboard",
        ["LSOA heat map · segments", "Green Profits (scenario)"], kind="amber")

    # vertical connectors between rows (centre spine + side spines)
    cxs = [xl + w3 / 2, xl + w3 + 8 + w3 / 2, xl + 2 * (w3 + 8) + w3 / 2]
    mid = xl + full / 2
    for i in range(7):
        top = ys[i]
        bot = ys[i + 1] + rh
        if i in (1, 3, 4):  # fan out / in to three columns
            for c in cxs:
                arrow(d, c, top, c, bot)
        elif i == 2:
            arrow(d, cxs[1], top, cxs[1], bot)
            arrow(d, cxs[0], top, cxs[0], bot)
            arrow(d, cxs[2], top, cxs[2], bot)
        elif i == 5:
            arrow(d, xl + wk / 2, top, xl + wk / 2, bot)
            arrow(d, xl + wk + 8 + wk / 2, top, xl + wk + 8 + wk / 2, bot)
        elif i == 6:
            for c in cxs:
                arrow(d, c, top, c, bot)
        else:
            arrow(d, mid, top, mid, bot)
    # horizontal links within the twin row
    arrow(d, xl + w3 + 8, ys[2] + rh / 2, xl + w3, ys[2] + rh / 2)
    arrow(d, xl + 2 * (w3 + 8) - 8, ys[2] + rh / 2, xl + 2 * (w3 + 8), ys[2] + rh / 2)

    # what-if loop on the right gutter: experience -> scenario engine
    gx = W - 10
    d.add(Line(xl + full, ys[7] + rh / 2, gx, ys[7] + rh / 2, strokeColor=BLUE, strokeWidth=1.1))
    d.add(Line(gx, ys[7] + rh / 2, gx, ys[5] + rh / 2, strokeColor=BLUE, strokeWidth=1.1))
    arrow(d, gx, ys[5] + rh / 2, xl + full, ys[5] + rh / 2, color=BLUE, width=1.1)
    for k, t in enumerate(["what-if", "reruns", "models"]):
        label(d, gx + 1, ys[6] + rh / 2 + 8 - k * 8, t, 6.0, BLUE, "middle", True)
    return d


# ----------------------------------------------------------------------------- diagram 2: block dependencies
def block_graph():
    W, H = 482, 300
    d = Drawing(W, H)
    bw, bh = 128, 46
    pos = {
        "B1": (0, 230), "B2": (177, 230), "B3": (354, 230),
        "B5": (0, 140), "B4": (177, 140),
        "B6": (177, 60),
        "B7": (60, 0), "B8": (294, 0),
    }
    meta = {
        "B1": ("B1  Data foundation", ["data, splits, factors"], "green"),
        "B2": ("B2  Twin + rating + synth", ["physics, labels"], "core"),
        "B3": ("B3  Bayesian + energy", ["posterior samples"], "core"),
        "B4": ("B4  Score + retrofits", ["ladder, Pareto"], "plain"),
        "B5": ("B5  Property value", ["Bayesian hedonic"], "plain"),
        "B6": ("B6  Scenario + API + LLM", ["run_scenario, chat"], "green"),
        "B7": ("B7  Customer app + 3D", ["web/"], "amber"),
        "B8": ("B8  Lloyds dashboard", ["geo + scenarios"], "amber"),
    }
    for k, (x, y) in pos.items():
        t, ls, kind = meta[k]
        box(d, x, y, bw, bh, t, ls, kind=kind, tsize=7.8)

    def c(k, side):
        x, y = pos[k]
        return {"r": (x + bw, y + bh / 2), "l": (x, y + bh / 2), "t": (x + bw / 2, y + bh),
                "b": (x + bw / 2, y)}[side]

    edges = [
        ("B1", "r", "B2", "l", "Property, Weather, factors"),
        ("B2", "r", "B3", "l", "simulate()"),
        ("B1", "b", "B5", "t", "SaleRecord, HPI"),
        ("B2", "b", "B4", "t", "apply_retrofits, rating"),
        ("B3", "b", "B4", "r", "posterior samples"),
        ("B5", "r", "B4", "l", ""),
        ("B4", "b", "B6", "t", "ScoreResult, RetrofitEffect, Pareto"),
        ("B5", "b", "B6", "l", "ValueImpact"),
        ("B6", "b", "B7", "t", "ScenarioResult (API)"),
        ("B6", "b", "B8", "t", "/dashboard/*"),
    ]
    for a, sa, b, sb, txt in edges:
        if a == "B5" and b == "B4":
            continue
        x1, y1 = c(a, sa)
        x2, y2 = c(b, sb)
        if a == "B3" and b == "B4":
            d.add(Line(x1, y1, x1, y2, strokeColor=LINE, strokeWidth=0.9))
            arrow(d, x1, y2, x2, y2)
            label(d, x1 - 4, (y1 + y2) / 2 + 6, txt, 6.2, GREY, "end")
            continue
        if a == "B5" and b == "B6":
            d.add(Line(x1, y1, x1, y2, strokeColor=LINE, strokeWidth=0.9))
            arrow(d, x1, y2, x2, y2)
            label(d, x1 + 4, y2 + 4, txt, 6.2, GREY, "start")
            continue
        arrow(d, x1, y1, x2, y2)
        if not txt:
            continue
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        if a == "B1" and b == "B2":
            label(d, mx, my + 13, "Property,", 6.0)
            label(d, mx, my + 5, "weather", 6.0)
        elif a == "B6" and b == "B7":
            label(d, mx - 6, my - 2, txt, 6.2, GREY, "end")
        elif a == "B6" and b == "B8":
            label(d, mx + 6, my - 2, txt, 6.2, GREY, "start")
        elif y1 == y2:
            label(d, mx, my + 4, txt, 6.2)
        else:
            label(d, mx + 4, my, txt, 6.2, GREY, "start")
    # B2 -> B5: estimated rating feeds the value model's "after" input
    x1, y1 = pos["B2"][0] + 20, pos["B2"][1]
    x2, y2 = pos["B5"][0] + bw, pos["B5"][1] + bh - 8
    arrow(d, x1, y1, x2, y2)
    label(d, (x1 + x2) / 2 + 4, (y1 + y2) / 2 + 6, "RatingEstimate", 6.0, GREY, "start")
    label(d, W, H - 6, "Arrows = contracts passed between blocks; every block can start on mocks.", 6.6, GREY, "end")
    return d


# ----------------------------------------------------------------------------- diagram 3: what-if flow
def whatif_flow():
    W, H = 482, 92
    d = Drawing(W, H)
    steps = [
        ("User", ["3D toggle or", "chat question"]),
        ("Router", ["Jev / Claude", "→ ScenarioSpec"]),
        ("Feasibility", ["rules veto", "→ Jev"]),
        ("run_scenario", ["twin on every", "posterior sample"]),
        ("ScenarioResult", ["Distributions", "+ provenance"]),
        ("LLM + validator", ["quotes tool", "numbers only"]),
        ("UI", ["numbers, 3D,", "explanation"]),
    ]
    n = len(steps)
    g = 8
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
def build():
    # ---------------------------------------------------------------- cover
    story.append(Spacer(1, 40 * mm))
    P("GreenTwin", "title")
    P("System design and decoupled build blocks", "subtitle")
    story.append(Spacer(1, 4 * mm))
    P("Megathon X: Singularity · Lloyds Problem Statement 2: Green Home Score and Retrofit Recommendation "
      "Engine · Team MegaMind", "meta")
    story.append(Spacer(1, 10 * mm))
    callout("<b>In one line.</b> A Bayesian-calibrated physics twin of every home, which you can see in 3D, "
            "explain in plain language and act on, where every number is backed by physics and the range it "
            "comes with.")
    story.append(Spacer(1, 6 * mm))
    P("<b>Audience:</b> the build team. This document fixes the architecture, the contracts between parts, "
      "and eight build blocks that can each be built, tested and documented independently. Design detail for "
      "individual techniques lives in <font face='%s'>docs/STRATEGY_REPORT.md</font>; binding rules live in "
      "<font face='%s'>AGENTS.md</font> and <font face='%s'>contracts/</font>, which win over this document if "
      "they ever disagree." % (F, F, F))
    story.append(Spacer(1, 6 * mm))
    table([
        ["Section", "What it covers"],
        ["1. Philosophies", "The six principles every block follows"],
        ["2. Architecture", "End-to-end system diagram, layers and the what-if loop"],
        ["3. Uncertainty", "How every number becomes a full distribution"],
        ["4. Build blocks", "Eight decoupled blocks, their interfaces and dependency graph"],
        ["5. Block specs", "Goal, build steps, models, outputs and done-when for each block"],
        ["6. Key mechanisms", "Retrofit engine + improvement ladder, energy prediction, property value"],
        ["7. Evaluation & submission", "Metrics vs baselines, PS mapping, submission items A–I"],
        ["8. Working rules", "Hackathon coding and UI guidelines, open decisions"],
    ], [45 * mm, 125 * mm], zebra=False)
    story.append(PageBreak())

    # ---------------------------------------------------------------- 1 philosophies
    H1("1. Core philosophies")
    table([
        ["#", "Philosophy", "What it means in the system"],
        ["1", "<b>Physics is the source of truth</b>",
         "Every energy, savings, carbon and rating number comes from the Level-1 digital twin. ML sets priors "
         "(EBM), estimates value and checks results; it never produces a customer's savings, so the system "
         "handles retrofit combinations no dataset has seen."],
        ["2", "<b>Bayesian in thought</b>",
         "Each home is calibrated against its own meter readings. Every output is a full distribution (mean, "
         "median, mode, std, P1–P99, 90% interval, tail probabilities) plus P(Poor), P(suitable), P(on front). "
         "Missing data never becomes zero; it widens the range."],
        ["3", "<b>Rerun, never add up</b>",
         "A retrofit is a change to the twin's parameters; every what-if and every package reruns the model. "
         "Interactions (insulation before a heat pump, electricity rising as gas falls) come from the physics. "
         "Savings are always per fuel."],
        ["4", "<b>Grounded explanation</b>",
         "The LLM can only call tools and quote what they return; a validator rejects any other number. Jev "
         "only routes questions and screens feasibility. No backing → \"unavailable\", never a guess."],
        ["5", "<b>Explainability first</b>",
         "Score waterfall, heat loss per building element, EBM contributions, value-model coefficients, and a "
         "reason on every recommendation or veto."],
        ["6", "<b>See it to believe it</b>",
         "A living 3D twin explored inside and out: heat loss glows, retrofits appear as they are added, and "
         "score, savings and bills update live, all from the same calibrated model."],
    ], [8 * mm, 42 * mm, 120 * mm])
    H2("The core rule from the brief, and how the design enforces it")
    P("<i>\"The score, savings, carbon and value figures come from documented models and calculations. The LLM "
      "only explains them... For a what-if request, the relevant models must first be rerun.\"</i> (PS §1)")
    bullets([
        "Only <b>B2/B3</b> (physics + Bayesian), <b>B4</b> (score formula, rules) and <b>B5</b> (value model) "
        "produce numbers. Each number carries <font face='%s'>provenance</font> naming its model." % F,
        "Every what-if goes through <b>run_scenario</b> (B6) before the LLM sees anything.",
        "The LLM's answer passes a <b>numeric validator</b>: any number not in that turn's tool output is "
        "rejected → regenerate once → template explainer.",
    ])
    story.append(PageBreak())

    # ---------------------------------------------------------------- 2 architecture
    H1("2. System architecture (end to end)")
    P("Data flows top to bottom through one physics twin. Everything below the twin consumes its posterior "
      "samples; the experience layer never computes, it renders <font face='%s'>ScenarioResult</font>. "
      "A what-if from the 3D twin or the chat loops back into the scenario engine (blue)." % F)
    story.append(architecture())
    story.append(Spacer(1, 4))
    P("Colour key: blue = physics/Bayesian core (the only source of energy numbers); green = data and "
      "orchestration; grey panels = LLM side (reads only through tools); amber = what users see.", "small")
    story.append(PageBreak())

    H2("Layers and their contracts")
    table([
        ["Layer", "Blocks", "Consumes", "Produces (contracts/)"],
        ["Inputs + data", "B1", "public datasets, customer input", "Property, EnergyUsageMonth, SaleRecord, "
         "WeatherMonth, factor tables, retrofit catalogue"],
        ["Physics", "B2", "Property, WeatherMonth, catalogue", "TwinParams, TwinOutput (per-fuel kWh, "
         "ElementHeatLoss), RatingEstimate, synthetic data + labels"],
        ["Bayesian + prediction", "B3", "TwinParams, usage, weather", "posterior samples; EnergyBreakdown "
         "(Distributions)"],
        ["Decisions + value", "B4, B5", "posterior samples, Property, sales", "ScoreResult, Feasibility, "
         "RetrofitEffect, ParetoPoint, ValueImpact"],
        ["Orchestration", "B6", "all of the above", "ScenarioResult (one per what-if), API, chat answers, "
         "WhatIfRoute, ExplanationCheck, CustomerMemory"],
        ["Experience", "B7, B8", "API + contracts/types.ts", "customer app, 3D twin, Lloyds dashboard"],
    ], [30 * mm, 16 * mm, 50 * mm, 74 * mm])
    H2("The what-if loop")
    story.append(whatif_flow())
    story.append(Spacer(1, 6))
    H2("Technology choices")
    table([
        ["Concern", "Choice"],
        ["Data + storage", "Python 3.12, pandas/Polars, DuckDB + Parquet (with meta.json per file)"],
        ["Physics twin", "numpy steady-state monthly RC model (Option A); differentiable PyTorch RC graph later "
         "behind the same simulate() (Option B)"],
        ["Bayesian", "NumPyro: NUTS offline for hierarchy, SVI per home (&lt; 1 s); ArviZ diagnostics"],
        ["Learned models", "EBM (InterpretML) primary; XGBoost baseline"],
        ["Value", "NumPyro hierarchical hedonic regression; OLS baseline"],
        ["API", "FastAPI; run_scenario cached by hash(spec + model versions)"],
        ["LLM", "Claude tool use + numeric validator + template fallback; Jev (System 1) if a key exists, "
         "else Claude structured output"],
        ["Front end", "React + Vite + TypeScript, react-three-fiber + drei (3D), Recharts, MapLibre + deck.gl, "
         "Zustand"],
    ], [32 * mm, 138 * mm])
    story.append(PageBreak())

    # ---------------------------------------------------------------- 3 uncertainty
    H1("3. Uncertainty: every number is a distribution")
    P("We never know a home's exact physics, so we keep many plausible versions of it (posterior samples) and "
      "run everything on all of them. Each output is computed <b>once per sample</b>, then summarised. "
      "Intervals are never added or multiplied together: the samples carry the correlations (a leaky home "
      "saves more from every retrofit).")
    H2("The Distribution contract (contracts/models.py, built by contracts/summarise.py)")
    table([
        ["Part", "Fields", "Why"],
        ["Centre", "mean · median · mode (peak of the sample density)", "They differ when the result is "
         "lopsided; showing all three stops the mean hiding it"],
        ["Spread", "standard deviation · IQR", "How uncertain the number is"],
        ["Percentiles", "P1, P5, P10, P25, P50, P75, P90, P95, P99", "The full shape, tails included"],
        ["Credible intervals", "50% (P25–P75) and 90% (P5–P95)", "The 90% interval is the headline range"],
        ["Tails", "labelled tail probabilities (e.g. P(bill rises &gt; £100/yr)) · mean of the worst 5% · "
         "skewness", "What could go wrong, and how badly"],
        ["Shape + meta", "histogram · unit · n_samples · headline (mean or median if skewed)",
         "Density plot in the UI; traceability"],
    ], [26 * mm, 74 * mm, 70 * mm])
    P("Categories (score band, EPC band, verdict) report the probability of every category plus the mode. "
      "Probabilities (P(suitable), P(on front)) are reported as they are. Only documented constants (PS "
      "weights, emission factors, tariffs, grants, rule vetoes) are bare numbers.", "small")
    H2("Where the uncertainty comes from")
    table([
        ["Source", "How it is modelled"],
        ["The home's physics (heat loss, setpoint, baseload, COP)", "Posterior from calibration on its meter "
         "readings; prior only (wider) without meter data"],
        ["Missing months, estimated reads, imputed fields", "Latent variables / wider noise, never 0"],
        ["Weather", "Typical, cold and warm years sampled"],
        ["Rebound after retrofit", "Prior on setpoint rising after fabric retrofits"],
        ["Retrofit costs", "Distribution over the sourced low–high range (BEIS 2017 × CPI)"],
        ["Rating after retrofit", "Twin samples through the SAP estimator → SAP distribution + band "
         "probabilities"],
        ["Property value", "Posterior of the hedonic model (location effects, rating effect)"],
        ["Feasibility", "Jev confidence recalibrated (isotonic) into P(feasible); rule veto = 0"],
    ], [62 * mm, 108 * mm])
    H2("Example: heat-pump bill change on SH2 (illustrative)")
    table([
        ["Mean", "Median", "Mode", "Std", "P5", "P25", "P75", "P95", "P(saving &gt; 0)", "P(rise &gt; £100)"],
        ["−£35", "−£42", "−£55", "£88", "−£180", "−£95", "+£20", "+£110", "0.38", "0.23"],
    ], [15 * mm, 15 * mm, 15 * mm, 14 * mm, 16 * mm, 16 * mm, 16 * mm, 16 * mm, 23 * mm, 24 * mm], zebra=False)
    P("The mean alone hides that about 1 in 4 homes like this could see the bill rise by over £100/yr, while "
      "carbon falls with near certainty. That is the brief's \"electricity rises after a heating change\" "
      "scenario, stated honestly.", "small")
    story.append(PageBreak())

    # ---------------------------------------------------------------- 4 blocks overview
    H1("4. Build blocks (decoupled parts of the system)")
    P("Eight blocks, each a complete piece one person can build, test and document end to end. Blocks talk "
      "<b>only</b> through <font face='%s'>contracts/</font> (Pydantic models + the TypeScript mirror), so "
      "every block starts on day one against <font face='%s'>contracts/examples/</font> and stubs." % (F, F))
    story.append(block_graph())
    story.append(Spacer(1, 6))
    H2("Interfaces: what each block exposes")
    table([
        ["Block", "Package", "Public interface (stub it first)", "Start on"],
        ["B1 Data", "greentwin/data, geo", "load_property(id), load_usage(id), load_weather(area), load_sales(), "
         "emission_factors(), tariffs(), catalogue()", "examples + small CSV samples"],
        ["B2 Twin", "greentwin/twin", "params_from_property(p), simulate(params, weather, mode), "
         "apply_retrofits(params, actions), estimate_rating(...), generate_synthetic(n, seed)", "RdSAP defaults"],
        ["B3 Bayesian", "greentwin/bayes, models", "intensity_prior(p) → Distribution, calibrate(p, usage, weather) → "
         "Posterior, predict(posterior, spec) → samples", "synthetic data from B2"],
        ["B4 Score + retrofits", "greentwin/score, retrofit, optimise", "score(p, energy) → ScoreResult, "
         "feasibility(p) → [Feasibility], ladder(p, posterior, goal) → [RetrofitEffect], pareto(...) → "
         "[ParetoPoint]", "recorded inputs; fake samples"],
        ["B5 Value", "greentwin/value", "value_impact(p, rating_after) → ValueImpact, green_twins(p)",
         "UCL sales sample"],
        ["B6 Orchestration", "greentwin/api, llm, jev, kb", "run_scenario(spec) → ScenarioResult; REST API; "
         "chat(...), route(...) → WhatIfRoute", "MOCK ScenarioResult"],
        ["B7 Customer app", "web/", "screens over the API using contracts/types.ts", "mock server"],
        ["B8 Dashboard", "greentwin/geo + web/", "area_layers(layer, region), segments(), "
         "green_profits(assumptions)", "B1 data + batch outputs"],
    ], [24 * mm, 30 * mm, 86 * mm, 30 * mm])
    story.append(PageBreak())

    # ---------------------------------------------------------------- 5 block specs
    H1("5. Block specifications")
    specs = [
        ("B1 · Data foundation", "PS §9 · feeds B2, B3, B4, B5, B8",
         "Clean, licensed, split, joinable real data with every external factor traceable to a cited source.",
         ["uv + Python 3.12 environment; CI runs pytest.",
          "Download scripts → Parquet + meta.json: EPC certificates + recommendations + certificate history "
          "(North West, South West, London), NEED 50k, UCL Price Paid–EPC linked sales + UK HPI, weather degree "
          "days (HadUK-Grid / Open-Meteo), ONS postcode → LSOA, IMD, off-gas, planning constraints, OSM roofs.",
          "Sourced factors in contracts/: DESNZ 2025 emissions, Ofgem tariffs, BEIS 2017 × CPI costs; unsourced "
          "items return null + reason.",
          "Mappers to Property / SaleRecord; splits.py (70/15/15 by customer, else property, seed 42, assigned "
          "once); leakage audit (epc.potential_* are labels only); load into DuckDB."],
         "Real data tables, factor tables, load_* helpers, docs/DATA.md",
         "No id spans two splits; no leakage fields in feature lists; every table has meta.json; DATA.md done."),
        ("B2 · Physics twin + rating estimator + synthetic data", "PS §5, §7, §9 · feeds B3, B4, B5, B7",
         "A documented physics model that turns any home and retrofit package into per-fuel energy and an "
         "estimated rating, and a generator that uses it to create realistic training data and labels.",
         ["Level-1 twin: element U·A from RdSAP defaults, ventilation, degree days, hot water, appliances; "
          "boiler efficiency / heat-pump COP → electricity + gas; PV cuts imports. Modes: calibrated and "
          "standard (SAP occupancy).",
          "apply_retrofits: each catalogue item is a parameter edit; packages are always re-simulated.",
          "Rating estimator (contracts/sap_estimator.json): SAP 2012 formula on the twin in standard mode, "
          "anchored to the recorded SAP; labelled \"estimated rating, not an EPC\".",
          "Synthetic generator: real EPC homes + NEED-calibrated behaviour noise → monthly usage with gaps and "
          "conflicts, plus reference labels (per-fuel savings, improvement potential, feasibility); planted "
          "value premium for method checks only."],
         "TwinParams, TwinOutput, ElementHeatLoss (estimated=true), RatingEstimate, synthetic datasets",
         "SH1–SH3 within ±15% of metered totals; monotonicity sweeps pass; synthetic vs NEED report; "
         "rating validated on EPC before/after pairs."),
        ("B3 · Bayesian calibration + energy prediction", "PS §3, §5, §6 · feeds B4, B6",
         "Fit each home's unknown physics to its own meter data and predict per-fuel energy with full "
         "distributions.",
         ["EBM energy-intensity prior trained on NEED (baselines: NEED median, XGBoost).",
          "NumPyro hierarchical calibration (home → LSOA → region, by type) of heat loss, setpoint, baseload, "
          "COP; missing months latent; NUTS offline, SVI per home.",
          "predict(): every posterior sample through apply_retrofits + simulate for baseline, package and each "
          "step in sequence; weather years sampled; rebound prior.",
          "Annualised metered intensity for the score (not weather-normalised); weather-normalised figure as a "
          "sensitivity only."],
         "Posterior samples, EnergyBreakdown of Distributions, data-confidence score",
         "Held-out 90% coverage ≈ 90%, CRPS, R-hat &lt; 1.01; beats degree-day OLS / LightGBM quantile; "
         "masked-month imputation error reported."),
        ("B4 · Green Home Score + retrofit engine", "PS §2, §3, §4 · feeds B6, B7, B8",
         "Answer \"how green is my home\" and \"what should I improve\" with feasible, ordered, explained "
         "recommendations.",
         ["Score from contracts/score_mapping.json: fixed weights, monotonic sub-scores, rounded, waterfall; "
          "score distribution and P(each band) from posterior samples; overlaps justified.",
          "Poor-home detector: P(Poor/Very poor), key issues per m² with peer context.",
          "Feasibility: rule vetoes → Jev (or Claude fallback) on survivors → calibrated P(feasible).",
          "Improvement ladder: twin reruns in fabric-first order; per-step distributions; P(suitable) = "
          "P(feasible) × P(net benefit) → High/Medium/Low; probabilistic verdicts.",
          "Pareto front over feasible packages with P(on front), knee and budget filter."],
         "ScoreResult, Feasibility, RetrofitEffect, ParetoPoint, docs/SCORE.md",
         "SH1 blocks EWI; SH2 shows electricity rising; SH3 blocks loft/PV and marks the heat pump "
         "needs_survey; ranking vs reference labels and vs real EPC recommendations reported."),
        ("B5 · Property value", "PS §7 · feeds B6, B7",
         "An indicative value range, before vs after, accounting for location, size, type and market.",
         ["Bayesian hierarchical hedonic model on real sales: log(price ÷ HPI) = location (LSOA within LA "
          "within region) + size + type + form + age + tenure + β(region × type) × rating.",
          "Same home at recorded vs estimated-after rating, applied to the current value estimate (never a "
          "past sale price); full distribution; wording always \"indicative\".",
          "Stretch checks: repeat sales, matched comparable homes (\"green twins of your home\")."],
         "ValueImpact (Distribution + wording), docs/VALUE.md",
         "Held-out coverage and error vs OLS without the rating; β vs Nationwide / DECC 2013; planted-premium "
         "recovery reported as a method check only."),
        ("B6 · Scenario engine + API + knowledge base + grounded LLM + memory", "PS §1, §8, §10 · feeds B7, B8",
         "One run_scenario that reruns every model, an API over it, and an LLM that only explains returned "
         "numbers.",
         ["run_scenario assembles ScenarioResult from B2–B5 with provenance; caching; mock mode for the UI.",
          "FastAPI endpoints: homes, scenario, feasibility, pareto, green-twins, chat, route, dashboard, memory.",
          "Knowledge base: SQL (numbers), vector store (docs, grants), embedded graph (rules, constraints).",
          "LLM: tools only, system-prompt rules, numeric validator, template fallback, cached calls; Jev router "
          "with isotonic calibration, Claude fallback.",
          "Customer memory updates constraints and posteriors; ≥ 30 adversarial LLM scenarios."],
         "ScenarioResult, REST API, chat answers with ExplanationCheck, docs/LLM.md",
         "Real results for SH1–SH3 validate; 0 unvalidated numbers; groundedness and accuracy ≥ 90%, "
         "unsupported claims &lt; 5%; works with no keys."),
        ("B7 · Customer app + 3D twin", "PS §2–§7, §10 · the demo",
         "The customer journey from \"how green is my home\" to a green-loan conversation.",
         ["Shared theme + components (NumberWithRange, DistributionCard, ScoreWaterfall, ProvenanceTag).",
          "Screens: score + waterfall, improvement ladder, what-if chat, 3D twin, Pareto + green loan, memory, "
          "agent capture v1.",
          "3D: parametric house from EPC fields; outside, cutaway and thermal views; retrofits appear; numbers "
          "update via run_scenario; precomputed packages for showcase homes."],
         "web/ app",
         "Full journey runs live on SH1–SH3 and offline from cache; demo script runs without errors."),
        ("B8 · Lloyds dashboard + geography", "PS §12 business relevance · the business story",
         "Turn home-level results into green-lending demand; every number computed or a labelled assumption.",
         ["Batch scoring over EPC regions → LSOA / district aggregates (Distributions summed over samples).",
          "Heat map layers: green share, Poor share, finance demand, carbon potential, grant eligibility, "
          "hotspots, degree-day trends.",
          "Segments; Green Profits scenario from contracts/green_profits_assumptions.json (assumption sliders, "
          "Monte Carlo); funnel. Synthetic customer base labelled as such."],
         "/dashboard/* API + dashboard screens",
         "Runs on three regions; every assumption visibly labelled."),
    ]
    for title, tag, goal, steps, outputs, done in specs:
        block = [
            Paragraph(title, S["h2"]),
            Paragraph("<font color='#5b5b5b'>%s</font>" % tag, S["small"]),
            Paragraph("<b>Goal.</b> " + goal, S["body"]),
        ]
        block += [Paragraph(s, S["bullet"], bulletText="\u2022") for s in steps]
        block.append(Paragraph("<b>Outputs.</b> " + outputs, S["body"]))
        block.append(Paragraph("<b>Done when.</b> " + done, S["body"]))
        story.append(CondPageBreak(60 * mm))
        story.append(KeepTogether(block))
    story.append(PageBreak())

    # ---------------------------------------------------------------- 6 mechanisms
    H1("6. Key mechanisms")
    H2("6.1 Retrofit engine: rules + Jev decide what is possible; the twin decides what it is worth")
    bullets([
        "<b>Feasibility:</b> hard rule vetoes first (conservation area → no external wall insulation, no roof "
        "access → no PV/loft, ...), then Jev on the survivors: feasible / with conditions / needs survey / "
        "infeasible with a calibrated confidence. Jev never produces a number.",
        "<b>Simulation:</b> step k's marginal effect = twin(steps 1..k) − twin(steps 1..k−1) on every posterior "
        "sample, in the order fabric → glazing → controls → heating → generation. The marginals add up exactly "
        "to the package, and the package is one twin run.",
        "<b>Verdicts (probabilistic):</b> Recommended if P(suitable) ≥ 0.8 and P(step adds ≥ 3% of purchased "
        "energy or ≥ 2 score points) ≥ 0.8; Optional / Not worth adding / Needs survey / Not feasible otherwise. "
        "Thresholds to be agreed; \"net benefit\" follows the customer's goal (bills or carbon).",
    ])
    P("<b>Improvement ladder, SH2 (illustrative; mean with 90% interval).</b> Baseline 15,200 kWh/yr purchased, "
      "score 60.", "body")
    table([
        ["#", "Add", "Gas saving kWh/yr", "Elec. saving kWh/yr", "Potential so far", "Score after",
         "CO2 t/yr", "Bill £/yr", "Verdict"],
        ["1", "Cavity wall insulation", "2,600 (2,100–3,100)", "0", "17%", "66", "0.48", "+207", "Recommended"],
        ["2", "Loft top-up to 270 mm", "900 (650–1,150)", "0", "23%", "68", "0.16", "+72", "Recommended"],
        ["3", "Draught proofing", "400 (200–600)", "0", "26%", "69", "0.07", "+32", "Optional"],
        ["4", "Smart thermostat + TRVs", "500 (300–750)", "0", "29%", "71", "0.09", "+40", "Recommended"],
        ["5", "Replace old double glazing", "450 (250–650)", "0", "32%", "72", "0.08", "+36", "Not worth adding"],
        ["6", "Air-source heat pump", "7,150 (6,500–7,800)", "−2,300 (rises)", "64%", "80", "0.86", "−35",
         "Depends on goal"],
        ["7", "Solar PV 3 kWp", "0", "1,100 (import cut)", "71%", "86", "0.22", "+290", "Recommended"],
        ["", "<b>Package (one run)</b>", "<b>12,000</b>", "<b>−1,200</b>", "<b>71%</b>", "<b>86</b>",
         "<b>1.96</b>", "<b>+641</b>", ""],
    ], [7 * mm, 31 * mm, 24 * mm, 21 * mm, 16 * mm, 13 * mm, 12 * mm, 13 * mm, 33 * mm])
    P("Carbon: DESNZ 2025 (gas 0.18296, electricity 0.19553 kgCO2e/kWh). Bills: Ofgem Q4 2026 (gas 7.97p, "
      "electricity 26.32p). The heat pump raises electricity and leaves the bill centred near zero at current "
      "prices; removing the gas supply also saves the ~£108/yr standing charge, reported separately.", "small")
    H2("6.2 Improvement potential (PS §3)")
    bullets([
        "100 × (E_baseline − E_after) / E_baseline on purchased energy (grid import + gas), E_after from "
        "<b>all feasible retrofits</b> run together; reported as a distribution with the per-fuel split.",
        "Needs-survey items are reported separately; potential (technical maximum) is not the customer's plan.",
        "Baseline 0 or missing → unavailable + reason. Homes are ranked by the posterior mean with "
        "P(top 10%).",
    ])
    H2("6.3 Energy prediction (B2 + B3)")
    bullets([
        "Physics computes heat loss × degree days → fuel via efficiency / COP; solar cuts imports.",
        "Homes without meter data: EBM prior (trained on NEED) sets the expected kWh/m², wider distribution.",
        "Homes with meter data: Bayesian calibration fits the twin to the home's own bills.",
        "Output: electricity and gas kWh/yr as full Distributions; checked on held-out homes against the NEED "
        "median and XGBoost.",
    ])
    story.append(CondPageBreak(45 * mm))
    H2("6.4 Property value (B5)")
    bullets([
        "log(price ÷ local HPI) = location (LSOA within LA within region, pooled) + floor area + type + form + "
        "age + tenure + β(region × type) × rating.",
        "Value impact per posterior sample = V × (exp(β × ΔSAP) − 1), with ΔSAP from the rating estimator → full "
        "distribution; if the range includes 0, say the data shows no clear effect.",
        "Always \"Indicative range based on historical associations in sales data; not a promised or proven "
        "increase.\" Never a reason to borrow.",
    ])
    story.append(PageBreak())

    # ---------------------------------------------------------------- 7 evaluation + PS mapping
    H1("7. Evaluation and submission")
    H2("Evaluation (held-out, property-level splits, simple baselines)")
    table([
        ["Output", "Metric", "Baseline", "Block"],
        ["Energy intensity / prediction", "MAE/MAPE per fuel, 90% coverage", "NEED median, XGBoost", "B3"],
        ["Calibrated per-fuel energy", "coverage ≈ 90%, CRPS, PIT, R-hat", "degree-day OLS, LightGBM quantile",
         "B3"],
        ["Missing-month imputation", "MAE on masked months", "seasonal-naive ×12", "B3"],
        ["Rating after retrofit", "MAE + coverage of SAP change on EPC before/after pairs",
         "median band shift per measure", "B2"],
        ["Score", "0 monotonicity violations; every heating type mapped", "—", "B4"],
        ["Improvement-potential ranking", "Spearman / NDCG vs reference labels", "rank by total kWh", "B4"],
        ["Recommendations", "precision@k vs real EPC assessor lists; 0 infeasible", "recommend all feasible",
         "B4"],
        ["Pareto", "hypervolume, % greedy picks dominated", "greedy kWh/£", "B4"],
        ["Value", "coverage, pinball loss on held-out sales; β vs published", "OLS without rating", "B5"],
        ["Probabilities", "calibration (a \"0.8\" happens ~80% of the time)", "—", "B3, B4, B6"],
        ["LLM", "groundedness ≥ 90%, accuracy ≥ 90%, unsupported &lt; 5%", "no validator", "B6"],
        ["Synthetic data", "KS / quantiles vs NEED", "—", "B2"],
    ], [38 * mm, 62 * mm, 48 * mm, 22 * mm])
    H2("Problem statement coverage")
    table([
        ["PS section", "Requirement", "Block"],
        ["§1", "Core rule; what-if reruns the models first", "B6"],
        ["§2", "Score: fixed weights, monotonic documented sub-scores, contributions, justified overlaps", "B4"],
        ["§3", "Poor homes, key issues, improvement potential (per fuel)", "B3 + B4"],
        ["§4", "Feasible, prioritised retrofits with suitability, saving, reason; catalogue documented",
         "B1 + B4"],
        ["§5", "Per-retrofit and package savings per fuel, interactions, consistent baseline", "B2 + B3 + B4"],
        ["§6", "Carbon from documented per-fuel factors", "B1 + B3"],
        ["§7", "Indicative value range, before vs after, location/size/type/market", "B2 + B5"],
        ["§8", "Expected outputs, pipeline, what-if flow", "B6 + B7"],
        ["§9", "Data, schema, splits, leakage, documented synthetic labels", "B1 + B2"],
        ["§10", "Customer memory changes recommendations", "B6 + B7"],
        ["§11", "Evaluation vs baselines, scenario tests, performance", "all"],
    ], [18 * mm, 128 * mm, 24 * mm])
    H2("Submission items (PS §13)")
    table([
        ["Item", "Content", "Block"],
        ["A", "Working prototype", "B7 + B6"],
        ["B", "Source, data scripts, run instructions", "all"],
        ["C", "Trained models + inference code (artifacts/)", "B2, B3, B4, B5"],
        ["D", "Sub-score mappings (docs/SCORE.md)", "B4"],
        ["E", "LLM prompts, model/API, evaluation (docs/LLM.md)", "B6"],
        ["F", "Architecture diagram (this document)", "B6 + B7"],
        ["G", "Evaluation report on the validation set", "all"],
        ["H", "Responsible AI note", "B6 + B1"],
        ["I", "Held-out test predictions + schema/label docs", "B1 + B3/B4/B5"],
    ], [14 * mm, 132 * mm, 24 * mm])
    story.append(PageBreak())

    # ---------------------------------------------------------------- 8 working rules
    H1("8. Working rules")
    H2("Coding guidelines: speed first")
    callout("<b>This is a hackathon.</b> Speed is the goal, not bug-free code. A working, good-looking demo beats "
            "a perfect module nobody sees.")
    bullets([
        "Build the thinnest end-to-end path first (SH1–SH3 through twin → score → retrofits → API → UI), then "
        "deepen. Never polish one block in isolation.",
        "Work against contracts/examples and mocks from day one; never wait for another block.",
        "Reuse libraries and code; copy-paste is fine; no premature abstractions. Hardcode and precompute where it "
        "helps the demo; cache every slow or paid call.",
        "Stuck for ~45 minutes → cut scope, use the fallback or ask. Commit small and often; keep the demo path "
        "working.",
        "Tests only where judges look: contracts validate, monotonic sub-scores, per-fuel signs, null improvement "
        "potential at baseline 0, the LLM validator, no mock in the demo.",
    ])
    H2("Non-negotiables, even when rushing")
    bullets([
        "Contracts in and out; provenance on every result; uncertain numbers are Distributions built with "
        "contracts.summarise.",
        "Missing ≠ 0; savings per fuel; packages rerun, never summed; the LLM never produces a number; value "
        "always \"indicative\"; no keys in git.",
    ])
    H2("UI guidelines: one look everywhere")
    bullets([
        "One design system (web/src/theme) and shared components: NumberWithRange, DistributionCard, "
        "ScoreWaterfall, ProvenanceTag, StatusBadge. Never restyle per page.",
        "Palette: green #0b6e4f primary, blue #2a6fdb accent, neutral greys; colour-blind-safe scales "
        "(viridis/cividis) for heat loss and maps; red only for increases and negative savings.",
        "One font family (Inter), 8 px spacing grid, cards, one chart style; the 3D twin uses the same palette.",
        "Every number: unit + headline + 90% range; tap → DistributionCard. Labels: estimate, projected, "
        "indicative, estimated from home type.",
        "Polish the demo journey first (score → ladder → 3D → loan → chat); works offline for showcase homes.",
    ])
    H2("Open decisions")
    table([
        ["Decision", "Options", "Recommendation"],
        ["Twin ambition", "A: numpy steady-state twin first · B: differentiable PyTorch RC graph first",
         "A for checkpoint 1; upgrade behind simulate() if on time"],
        ["Verdict thresholds", "gain threshold (3% / 2 points), probability cut-offs, cost per kWh cut-off",
         "Agree once; record in docs/SCORE.md"],
        ["Default goal", "bills or carbon for \"net benefit\"", "Bills by default; carbon toggle"],
    ], [32 * mm, 80 * mm, 58 * mm])
    H2("Checkpoints")
    table([
        ["Checkpoint", "Done when"],
        ["1. End-to-end skeleton", "SH1–SH3 flow through real twin → score → retrofits → API → UI"],
        ["2. Complete submission", "Bayesian distributions, rating estimator, value, grounded LLM, evaluation "
         "report, docs D–I"],
        ["3. Demo build (freeze)", "3D twin, Pareto + loan, Lloyds dashboard, memory"],
    ], [42 * mm, 128 * mm])


def footer(c, doc):
    c.saveState()
    c.setFont(F, 7.5)
    c.setFillColor(GREY)
    c.drawString(20 * mm, 10 * mm, "GreenTwin · Team MegaMind · Megathon X, Lloyds PS2 · System design")
    c.drawRightString(A4[0] - 20 * mm, 10 * mm, f"{doc.page}")
    c.restoreState()


def main():
    build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=16 * mm, bottomMargin=18 * mm,
                            title="GreenTwin — System design and build blocks", author="Team MegaMind")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
