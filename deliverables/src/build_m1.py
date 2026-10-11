"""Build deliverables/M1.pdf — GreenTwin approach and idea (Megathon X, Lloyds PS2).

Run:  python deliverables/src/build_m1.py
Needs reportlab. Uses Segoe UI (Windows) or DejaVu Sans (Linux), else Helvetica.
"""

from pathlib import Path

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "deliverables" / "M1.pdf"

# ----------------------------------------------------------------------------- fonts
def _fonts():
    """Segoe UI on Windows, DejaVu Sans on Linux (both cover arrows, >= and GBP), else Helvetica."""
    for d, reg, bold, ital in (("C:/Windows/Fonts", "segoeui.ttf", "segoeuib.ttf", "segoeuii.ttf"),
                               ("/usr/share/fonts/truetype/dejavu", "DejaVuSans.ttf", "DejaVuSans-Bold.ttf",
                                "DejaVuSans-Oblique.ttf")):
        try:
            pdfmetrics.registerFont(TTFont("Seg", str(Path(d) / reg)))
            pdfmetrics.registerFont(TTFont("SegB", str(Path(d) / bold)))
            pdfmetrics.registerFont(TTFont("SegI", str(Path(d) / ital)))
            pdfmetrics.registerFontFamily("Seg", normal="Seg", bold="SegB", italic="SegI", boldItalic="SegB")
            return "Seg", "SegB"
        except Exception:  # pragma: no cover - try the next family
            continue
    return "Helvetica", "Helvetica-Bold"


F, FB = _fonts()

GREEN = colors.HexColor("#0b6e4f")
GREEN_L = colors.HexColor("#e6f2ec")
BLUE = colors.HexColor("#2a6fdb")
BLUE_L = colors.HexColor("#e7effc")
GREY = colors.HexColor("#5b5b5b")
GRID = colors.HexColor("#c9c9c9")
INK = colors.HexColor("#1d1d1d")

S = {
    "title": ParagraphStyle("title", fontName=FB, fontSize=30, leading=36, textColor=GREEN),
    "subtitle": ParagraphStyle("subtitle", fontName=F, fontSize=14, leading=20, textColor=INK),
    "meta": ParagraphStyle("meta", fontName=F, fontSize=10.5, leading=15, textColor=GREY),
    "h1": ParagraphStyle("h1", fontName=FB, fontSize=16, leading=21, textColor=GREEN, spaceBefore=6, spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName=FB, fontSize=11.5, leading=15, textColor=INK, spaceBefore=8, spaceAfter=3),
    "body": ParagraphStyle("body", fontName=F, fontSize=9.6, leading=13.6, textColor=INK, spaceAfter=4),
    "lead": ParagraphStyle("lead", fontName=F, fontSize=10.6, leading=15, textColor=INK, spaceAfter=6),
    "bullet": ParagraphStyle("bullet", fontName=F, fontSize=9.6, leading=13.4, leftIndent=12, bulletIndent=2, spaceAfter=1.5),
    "cell": ParagraphStyle("cell", fontName=F, fontSize=8.6, leading=11.2, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName=FB, fontSize=8.6, leading=11.2, textColor=INK),
    "small": ParagraphStyle("small", fontName=F, fontSize=8, leading=10.5, textColor=GREY),
    "callout": ParagraphStyle("callout", fontName=F, fontSize=9.8, leading=14, textColor=INK),
}

story: list = []


def P(text, style="body"):
    story.append(Paragraph(text, S[style]))


def H1(text):
    story.append(Paragraph(text, S["h1"]))


def H2(text):
    story.append(Paragraph(text, S["h2"]))


def bullets(items):
    for it in items:
        story.append(Paragraph(it, S["bullet"], bulletText="\u2022"))


def table(rows, widths, header=True, zebra=False):
    data = [[Paragraph(str(c), S["cellb"] if header and i == 0 else S["cell"]) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [
        ("GRID", (0, 0), (-1, -1), 0.4, GRID),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), GREEN_L))
    if zebra:
        for i in range(1, len(rows)):
            if i % 2 == 0:
                st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f7f7f7")))
    t.setStyle(TableStyle(st))
    story.append(t)
    story.append(Spacer(1, 6))


def callout(text):
    t = Table([[Paragraph(text, S["callout"])]], colWidths=[170 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), GREEN_L),
        ("LINEBEFORE", (0, 0), (0, -1), 3, GREEN),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))


# ----------------------------------------------------------------------------- diagrams
def box(d, x, y, w, h, title, lines=(), accent=False, fill=None):
    d.add(Rect(x, y, w, h, rx=5, ry=5, strokeColor=BLUE if accent else colors.HexColor("#9a9a9a"),
               strokeWidth=1.6 if accent else 0.9, fillColor=fill or (BLUE_L if accent else colors.white)))
    cy = y + h - 13
    d.add(String(x + w / 2, cy, title, fontName=FB, fontSize=8.6, fillColor=INK, textAnchor="middle"))
    for i, ln in enumerate(lines):
        d.add(String(x + w / 2, cy - 11 - i * 9.5, ln, fontName=F, fontSize=7.2, fillColor=GREY, textAnchor="middle"))


def arrow(d, x1, y1, x2, y2):
    d.add(Line(x1, y1, x2, y2, strokeColor=colors.HexColor("#8a8a8a"), strokeWidth=0.9))
    # arrow head (vertical or horizontal only)
    if x1 == x2:
        s = -1 if y2 < y1 else 1
        d.add(Polygon([x2, y2, x2 - 3, y2 - 5 * s, x2 + 3, y2 - 5 * s], fillColor=colors.HexColor("#8a8a8a"), strokeColor=None))
    else:
        s = -1 if x2 < x1 else 1
        d.add(Polygon([x2, y2, x2 - 5 * s, y2 - 3, x2 - 5 * s, y2 + 3], fillColor=colors.HexColor("#8a8a8a"), strokeColor=None))


def bus(d, xs, y):
    d.add(Line(min(xs), y, max(xs), y, strokeColor=colors.HexColor("#8a8a8a"), strokeWidth=0.9))


def architecture():
    W, H = 482, 430
    d = Drawing(W, H)
    cw, gap = 150, 16
    cx = [0, cw + gap, 2 * (cw + gap)]
    mid = [c + cw / 2 for c in cx]
    # rows (top -> bottom), y is bottom of box
    yA, yB, yC, yD, yE, yF = 360, 282, 204, 112, 52, 0
    bh = 56
    box(d, cx[0], yA, cw, bh, "Property & energy data", ["EPC, NEED, Land Registry", "UCL EPC-PPD link, sales"])
    box(d, cx[1], yA, cw, bh, "Weather & climate", ["HadUK-Grid, Open-Meteo", "PVGIS, UKCP18"])
    box(d, cx[2], yA, cw, bh, "Geography", ["ONS postcode -> LSOA, OSM", "conservation, listed, off-gas"])
    box(d, cx[0], yB, cw, bh, "Synthetic generator", ["sample homes -> run twin", "labels + behaviour noise"])
    box(d, cx[1], yB, cw, bh, "Level 1 physics twin", ["RC element model / graph", "retrofit = parameter edit"], accent=True)
    box(d, cx[2], yB, cw, bh, "Bayesian calibration", ["NumPyro, hierarchy", "posterior samples, 90% CI"])
    box(d, cx[0], yC, cw, bh, "Learned models", ["EBM (XGBoost baseline)", "value: quantile, matching, DR"])
    box(d, cx[1], yC, cw, bh, "Retrofit engine", ["rules + Jev feasibility", "Pareto, P(on front), staging"])
    box(d, cx[2], yC, cw, bh, "Geography & Lloyds", ["LSOA maps, hotspots", "segments, Green Profits"])
    # knowledge base container
    d.add(Rect(0, yD, W, 70, rx=5, ry=5, strokeColor=colors.HexColor("#9a9a9a"), strokeWidth=0.9,
               fillColor=colors.HexColor("#f2f2ee")))
    d.add(String(8, yD + 58, "Knowledge base (the LLM reads it only through tools)", fontName=FB, fontSize=8.4, fillColor=INK))
    kw = 146
    for i, (t, l) in enumerate([("SQL: numbers", "facts, results, posteriors"),
                                ("Vector DB: text", "methods, guidance, grants"),
                                ("Graph: relations", "rules, constraints, regions")]):
        box(d, 8 + i * (kw + 12), yD + 6, kw, 42, t, [l])
    box(d, 110, yE, 262, 46, "Grounded LLM + Jev System-1 router", ["tools + run_scenario, numeric validator"])
    for i, t in enumerate(["3D digital twin", "Customer app & chat", "Lloyds dashboard"]):
        box(d, cx[i], yF, cw, 36, t)
    # connectors
    for m in mid:
        d.add(Line(m, yA, m, yA - 10, strokeColor=colors.HexColor("#8a8a8a"), strokeWidth=0.9))
        arrow(d, m, yA - 10, m, yB + bh)
    bus(d, mid, yA - 10)
    arrow(d, cx[1], yB + bh / 2, cx[0] + cw, yB + bh / 2)
    arrow(d, cx[2], yB + bh / 2, cx[1] + cw, yB + bh / 2)
    d.add(Line(mid[1], yB, mid[1], yB - 11, strokeColor=colors.HexColor("#8a8a8a"), strokeWidth=0.9))
    bus(d, mid, yB - 11)
    for m in mid:
        arrow(d, m, yB - 11, m, yC + bh)
    for m in mid:
        d.add(Line(m, yC, m, yC - 11, strokeColor=colors.HexColor("#8a8a8a"), strokeWidth=0.9))
    bus(d, mid, yC - 11)
    arrow(d, mid[1], yC - 11, mid[1], yD + 70)
    arrow(d, mid[1], yD, mid[1], yE + 46)
    d.add(Line(mid[1], yE, mid[1], yE - 8, strokeColor=colors.HexColor("#8a8a8a"), strokeWidth=0.9))
    bus(d, mid, yE - 8)
    for m in mid:
        arrow(d, m, yE - 8, m, yF + 36)
    return d


def two_level():
    W, H = 482, 150
    d = Drawing(W, H)
    box(d, 120, 92, 242, 54, "Level 1: RC + equipment simulator", ["Bayesian-calibrated on metered energy",
                                                                   "source of every number; fallback"], accent=True)
    box(d, 0, 22, 225, 46, "Bayesian simulation + Pareto", ["posterior samples of Level 1",
                                                             "DeepONet surrogate only at portfolio scale"])
    box(d, 257, 22, 225, 46, "Level 2: visual only (optional)", ["surface heat-loss map from L1 fluxes",
                                                                  "spatial PINO; gated within +/-5% of L1"])
    d.add(Line(200, 92, 200, 80, strokeColor=colors.HexColor("#8a8a8a"), strokeWidth=0.9))
    d.add(Line(112, 80, 200, 80, strokeColor=colors.HexColor("#8a8a8a"), strokeWidth=0.9))
    arrow(d, 112, 80, 112, 68)
    d.add(Line(282, 92, 282, 80, strokeColor=colors.HexColor("#8a8a8a"), strokeWidth=0.9))
    d.add(Line(282, 80, 370, 80, strokeColor=colors.HexColor("#8a8a8a"), strokeWidth=0.9))
    arrow(d, 370, 80, 370, 68)
    d.add(String(112, 4, "Scores, savings, carbon, value, decisions", fontName=F, fontSize=7.6, fillColor=GREY, textAnchor="middle"))
    d.add(String(370, 4, "Illustrative 3D heat-flow view (never feeds numbers back)", fontName=F, fontSize=7.6, fillColor=GREY, textAnchor="middle"))
    return d


CONFIDENCE = [
    ("EBM + XGBoost baseline", "Must", .90, .78, .99), ("Physics twin (Level 1)", "Must", .85, .71, .98),
    ("Physics synthetic data", "Must", .85, .71, .98), ("Knowledge base + grounded LLM", "Must", .80, .65, .95),
    ("Bayesian calibration", "Must", .75, .58, .92), ("Geography analysis", "Should", .80, .65, .95),
    ("Pareto front + P(on front)", "Should", .80, .65, .95), ("Lloyds dashboard + Green Profits", "Should", .75, .58, .92),
    ("Green-twin matching (value)", "Should", .70, .52, .88), ("DR-learner value uplift", "Should", .65, .47, .83),
    ("3D twin UI", "Should", .60, .41, .79), ("Customer memory", "Could", .75, .58, .92),
    ("Jev routing + feasibility", "Could", .55, .36, .74), ("Neural-operator surrogate", "Could", .55, .36, .74),
    ("Photo-to-twin vision (v2)", "Could", .40, .22, .58), ("Spatial PINO heat-flow visual", "Could", .38, .19, .57),
]


def confidence_chart():
    rowh = 14
    W, H = 482, 30 + rowh * len(CONFIDENCE)
    d = Drawing(W, H)
    x0, x1 = 185, 430
    X = lambda p: x0 + p * (x1 - x0)  # noqa: E731
    top = H - 22
    for t in [0, .25, .5, .75, 1]:
        d.add(Line(X(t), 4, X(t), top + 4, strokeColor=colors.HexColor("#e2e2e2"), strokeWidth=0.6))
        d.add(String(X(t), top + 8, f"{int(t*100)}%", fontName=F, fontSize=7, fillColor=GREY, textAnchor="middle"))
    d.add(String(448, top + 8, "Tier", fontName=FB, fontSize=7, fillColor=GREY))
    prev = None
    for i, (name, tier, p, lo, hi) in enumerate(CONFIDENCE):
        y = top - 8 - i * rowh
        if prev and prev != tier:
            d.add(Line(0, y + rowh / 2, W, y + rowh / 2, strokeColor=colors.HexColor("#bdbdbd"), strokeWidth=0.5))
        prev = tier
        col = BLUE if tier == "Must" else colors.HexColor("#9c9c9c")
        d.add(String(x0 - 6, y - 2.5, name, fontName=F, fontSize=7.6, fillColor=INK, textAnchor="end"))
        d.add(Line(X(lo), y, X(hi), y, strokeColor=col, strokeWidth=1.6, strokeOpacity=0.5))
        d.add(Rect(X(p) - 2.6, y - 2.6, 5.2, 5.2, fillColor=col, strokeColor=None))
        d.add(String(448, y - 2.5, tier, fontName=F, fontSize=7.4, fillColor=INK))
    return d


# ----------------------------------------------------------------------------- content
def build():
    # Cover
    story.append(Spacer(1, 40 * mm))
    P("GreenTwin", "title")
    story.append(Spacer(1, 4))
    P("A calibrated physics digital twin for every UK home: Green Home Score, retrofit path, "
      "per-fuel savings, carbon and value, each with Bayesian confidence, and green home lending for Lloyds", "subtitle")
    story.append(Spacer(1, 10))
    P("Megathon X: Singularity · Problem Statement 2 (Lloyds): Green Home Score and Retrofit Recommendation Engine", "meta")
    P("Team MegaMind · Milestone 1 submission: approach and idea · 10 October 2026", "meta")
    story.append(Spacer(1, 18))
    callout("<b>In one sentence.</b> We model each home's physics, calibrate it with Bayesian inference against real metered "
            "energy, and use that one documented model to compute every number the customer sees. The LLM only explains "
            "those numbers. A 3D twin lets the customer add any retrofit and watch the effect, and a Lloyds dashboard "
            "turns this into targeted green-lending demand.")
    story.append(PageBreak())

    # 1 Executive summary
    H1("1. Executive summary")
    P("Customers want greener homes but do not know where to start; Lloyds wants to lend for retrofits but needs to know "
      "who will benefit. GreenTwin answers the brief's four questions (how green is my home, what should I improve, how "
      "much energy and carbon could I save, could it raise my value) with documented models, and adds what a bank needs to act.", "lead")
    bullets([
        "<b>One source of truth.</b> A Level-1 RC (resistance-capacitance) physics twin of each home, calibrated with "
        "Bayesian inference against metered gas and electricity, produces the score inputs, per-fuel savings, carbon, "
        "improvement potential and the inputs to the value model. Retrofits are parameter edits, so interactions "
        "(insulation before a heat pump) are computed, never summed.",
        "<b>Bayesian confidence everywhere.</b> Every number has a 90% credible interval; every retrofit has "
        "P(suitable) = P(feasible) × P(net benefit); every package on the Pareto front has P(on front).",
        "<b>Explainable score.</b> Fixed weights from the brief with monotonic, documented sub-score mappings and a "
        "contribution waterfall; an EBM (glass-box) model explains learned effects, with XGBoost as baseline.",
        "<b>Grounded GenAI.</b> A tool-only LLM that cannot calculate: what-ifs rerun the models, and a numeric validator "
        "rejects any number not in the computed output. Jev, a fast System-1 model, routes questions and screens "
        "feasibility with calibrated confidence; rule vetoes always win.",
        "<b>Value without overclaiming.</b> Real EPC-linked sale prices; a hedonic quantile model, a green-twin "
        "matched-pair study and a doubly robust (DR) learner; always a range, always labelled indicative.",
        "<b>Business plan.</b> A free customer-acquisition app plus agent visits that build the twin on site; a Lloyds "
        "dashboard with a locality heat map, segments and a Green Profits view of lending, retention and acquisition.",
        "<b>Experience.</b> A Three.js twin of the house, outside and inside, where every retrofit appears on the "
        "model and the computed numbers update, plus a staged retrofit timeline.",
    ])
    H2("How the design maps to the judging criteria")
    table([
        ["Criterion (weight)", "What GreenTwin contributes"],
        ["ML and recommendation quality (20%)", "Real public data + physics-generated labels validated against NEED; property-level splits; baselines for every model; feasible, ordered recommendations"],
        ["Green Home Score and insight (15%)", "Fixed weights, monotonic documented mappings, contribution waterfall, score interval and band probabilities"],
        ["Savings, carbon and value (15%)", "Per-fuel savings from the calibrated twin with interactions; documented emission factors; value range from three methods with causal caveats"],
        ["GenAI and LLM (15%)", "Tool-only LLM, rerun-on-what-if, numeric validator, 30–100 adversarial scenarios against the ≥90% / &lt;5% targets"],
        ["Business relevance (10%)", "Green-loan journey, agent and self-serve channels, Lloyds heat map, segments, Green Profits"],
        ["Architecture (10%)", "One contract layer, cached deterministic runs, seeds and data versions, offline demo"],
        ["Explainability and Responsible AI (5%)", "Glass-box score and EBM, provenance on every number, privacy by design, human adviser for credit"],
        ["UX (5%) and innovation (5%)", "Interactive 3D twin inside and outside; customer memory; Jev System-1/System-2 routing"],
    ], [52 * mm, 118 * mm], zebra=True)
    story.append(PageBreak())

    # 2 Business plan
    H1("2. Business plan: Lloyds green home lending")
    P("GreenTwin is a free customer-acquisition and green-lending tool. It is built on sections 2, 3, 4, 7 and 10 of the "
      "problem statement and turns each into a step in a lending conversation.", "lead")
    table([
        ["Brief section", "Role in the business"],
        ["2 · Green Home Score", "Shareable headline score that starts every conversation"],
        ["3 · Poorly performing homes", "Finds homes and areas with the most improvement potential, i.e. loan demand"],
        ["4 · Retrofit recommendations", "The prioritised feasible package becomes the loan's purpose and amount"],
        ["7 · Property value", "Indicative value range, based on historical associations in sales data, shown for information only, never as a promised increase or a reason to borrow"],
        ["10 · Customer memory", "Keeps each customer's plan, constraints and responses so follow-ups improve"],
    ], [52 * mm, 118 * mm])
    H2("Two channels")
    bullets([
        "<b>Agent-assisted.</b> A Lloyds agent visits, photographs the floor plan and rooms, and the app builds the twin "
        "on the spot (v1: photos plus manual element tags confirmed by the agent; v2: floor-plan parsing and element "
        "tagging with confidence scores). The customer sees the score, the retrofit path on the 3D house and what a "
        "green loan would mean.",
        "<b>Self-serve.</b> The customer looks up their EPC by postcode, uploads a home profile and explores improvements; "
        "a \"talk to an adviser\" button turns interest into a lead.",
    ])
    H2("Customer journey")
    P("Score and key issues → recommended retrofit path on the 3D house → package cost minus grants = indicative loan "
      "amount → monthly repayment vs monthly bill saving (\"cash-flow positive?\") → value range → save the plan or book "
      "an adviser. Loan figures are illustrative, not an offer or advice; any credit decision follows Lloyds' regulated "
      "process with a human adviser.")
    H2("Lloyds dashboard")
    bullets([
        "<b>Locality heat map (centrepiece)</b> by LSOA or postcode district with switchable layers: share of green homes "
        "(score ≥ 65), share of Poor/Very poor homes, customers interested in going green (what-ifs run, plans saved, "
        "adviser requests), estimated retrofit finance demand, carbon reduction potential, heat-pump grant eligibility "
        "(off-gas-grid) and density of existing customers.",
        "<b>Segments</b>, e.g. \"1930s semi, gas boiler, owner-occupier, low loan-to-value\", with size, average loan need, "
        "take-up from a documented scenario rule (every coefficient a labelled assumption on a slider; no model trained "
        "on synthetic labels) and carbon potential, ranked for outreach.",
        "<b>Green Profits view</b>: profit from green lending (balances × net interest margin, plus fees), retention "
        "(lower churn among customers who retrofit, valued as customer lifetime value) and acquisition (app users who "
        "become customers, net of acquisition and visit costs), with financed-emissions reduction alongside. Every "
        "assumption is a slider; results are Monte Carlo ranges.",
        "<b>Funnel</b>: app users → home profiles → saved plans → visits → applications → loans drawn.",
    ])
    P("The brief forbids real customer data, so the dashboard runs on a synthetic Lloyds customer base laid over real "
      "geography and real EPC distributions, clearly labelled. The idea that green homes carry lower credit risk is "
      "presented as a hypothesis, not a finding.", "small")
    story.append(PageBreak())

    # 3 Architecture
    H1("3. Solution architecture")
    P("Data and environment feed one physics twin; calibration, learned models and analysis build on it; all results "
      "land in a knowledge base that the LLM can reach only through tools.", "lead")
    story.append(architecture())
    story.append(Spacer(1, 6))
    P("A what-if from the 3D twin or the chat becomes a <b>ScenarioSpec</b> (retrofits, sizes, weather, samples). "
      "<b>run_scenario</b> edits the twin's parameters, propagates the posterior samples and the downstream models, and "
      "returns a <b>ScenarioResult</b> with provenance, which the UI renders and the LLM explains. Every module "
      "exchanges the same versioned data contracts (Pydantic models with a TypeScript mirror).")
    H2("Two-level simulation")
    story.append(two_level())
    P("Level 1 produces every number and drives every decision. Level 2 only visualises, and only when its integrated "
      "heat flow agrees with Level 1; information never flows back.", "small")

    # 4 Score
    H1("4. Green Home Score")
    P("S = Σ w<sub>k</sub> s<sub>k</sub> with the brief's fixed weights; each sub-score s<sub>k</sub> is 0–100 and "
      "monotonic (a better state never gives a lower sub-score, enforced by unit tests). The final score is rounded "
      "before classification.", "lead")
    table([
        ["Component (weight)", "Input", "Mapping (summary)"],
        ["Energy efficiency (25%)", "Recorded EPC SAP score / band; after a retrofit, the documented rating estimator", "SAP 1–100 directly; band midpoints A 96 … G 10 if only the band is known"],
        ["Energy intensity (20%)", "Annualised metered energy (grid import + gas) per m², as the brief defines", "Piecewise linear: 50 kWh/m² → 100, 150 → 70, 250 → 40, 400 → 0"],
        ["Insulation (15%)", "Roof, walls, floor", "Roof 40% / walls 45% / floor 15%, re-normalised for flats; loft depth 0–270 mm → 0–100; solid uninsulated 0 … as-built 100"],
        ["Heating system (15%)", "Main system and age", "Ground-source HP 100, air-source HP 95, district 70, condensing gas 55, communal boiler 50, high-heat-retention storage 50, electric storage 35, non-condensing gas 30, direct electric 30, LPG 25, oil 20, solid fuel 5; boilers over 12 years lose 1 point a year"],
        ["Windows and glazing (10%)", "Glazing types by share", "Single 0, secondary 50, double pre-2002 60, post-2002 80, triple 100"],
        ["Renewable energy (10%)", "PV kWp, battery, solar thermal", "0–5 kWp → 0–100, plus small bonuses, capped"],
        ["Smart energy management (5%)", "Controls present", "None 0 → programmer + thermostat 40 → smart + TRVs 90 → smart + zoning + smart meter 100"],
    ], [42 * mm, 46 * mm, 82 * mm], zebra=True)
    P("<b>Intensity definition.</b> Energy intensity is annualised metered energy (grid import plus gas) per m², exactly "
      "as the brief defines it. Months are annualised from those available using typical monthly shares; missing months "
      "are never zero; with no metered data the intensity is marked unavailable and the twin-modelled value is used with "
      "a warning. A weather-normalised intensity is reported only as a sensitivity. Using kWh/m² stops large homes being "
      "classed as inefficient for size alone.")
    P("<b>Inputs that affect two components.</b> Energy efficiency reflects the recorded rating (a modelled overall "
      "outcome); insulation, heating, glazing, renewables and controls are the levers a customer can change, so their "
      "overlap with the rating is accepted and documented. Solar PV affects two components: renewable energy credits "
      "installed generation, and intensity falls because self-consumed solar reduces metered import. We keep both "
      "because the brief defines intensity on metered energy and renewables as a separate lever: intensity measures what "
      "the home buys, renewables rewards generation the meter does not see. The evaluation reports the score with and "
      "without self-consumption added back, so the size of the overlap is visible.")
    P("<b>Post-retrofit rating.</b> A retrofitted home has no new EPC, so the projected score and the value model's "
      "band after retrofit use a documented estimator. The twin is run under standard conditions before and after the "
      "package, energy is priced with current Ofgem rates, and the SAP 2012 energy-cost-factor formula "
      "(ECF = 0.42 × cost ÷ (floor area + 45); SAP = 117 − 121·log<sub>10</sub>ECF, or 100 − 13.95·ECF below ECF 3.5) "
      "gives a modelled change that is added to the recorded SAP. Posterior samples give a rating interval and band "
      "probabilities. It is always labelled 'estimated rating, not an EPC', and is validated on homes with repeat EPCs.")
    P("<b>Classification:</b> 80–100 Highly efficient · 65–79 Good · 50–64 Moderate · 30–49 Poor · 0–29 Very poor. "
      "With Bayesian inputs we also report the score interval and band probabilities, e.g. 38 (90% CI 33–44), P(Poor) = 0.81. "
      "Poor and Very poor homes are flagged with their key issues and ranked by improvement potential "
      "100 × (E<sub>baseline</sub> − E<sub>after</sub>) / E<sub>baseline</sub> on purchased energy, reported per fuel; "
      "if the baseline is zero or missing the metric is shown as unavailable with the reason.")

    # 5 Twin
    H1("5. Physics digital twin and synthetic data")
    bullets([
        "<b>Level 1 twin (v1).</b> Steady-state RC element model: heat loss = Σ U·A + ventilation, driven by regional "
        "degree days; space heat and hot water converted to fuels by boiler efficiency or heat-pump SCOP; PV yield from "
        "PVGIS reduces grid import. Outputs per-fuel kWh and heat loss per envelope element (which colours the 3D twin).",
        "<b>v2 (stretch, GPU).</b> A differentiable RC graph (4–6 thermal nodes) whose conductances are predicted from EPC "
        "element features and solved as an ODE; energy conservation and monotonicity hold by construction.",
        "<b>Synthetic generator.</b> Samples homes from EPC / English Housing Survey joint distributions, attaches real "
        "weather, runs the twin and adds behaviour noise from NEED-calibrated priors. It produces monthly usage, "
        "per-retrofit reference labels and (for synthetic homes) property values from a hedonic model fitted on real "
        "sales plus an explicit green-premium term we choose. Recovering that premium is reported only as a method check, "
        "never as evidence of real uplift. All rules are disclosed; agreement with synthetic targets is not claimed as "
        "real-world accuracy, and the generator is validated against real NEED distributions.",
        "<b>Extrapolation</b> (new retrofit combinations, future climate) is done only by the physics twin; an "
        "out-of-distribution flag warns when a home is far from the training data.",
    ])
    # 6 Bayesian
    H1("6. Bayesian confidence in every output")
    P("NumPyro calibrates each home's heat-loss coefficient, heating setpoint, baseload and heat-pump efficiency against "
      "its metered monthly energy, with hierarchical priors by property type × region (RdSAP default U-values, "
      "Electrification of Heat trial SCOP). Missing months are latent, never zero. Fits are offline; per-home updates "
      "use variational inference in under a second.", "lead")
    table([
        ["Output", "Shown as", "Computed from"],
        ["Green Home Score", "38 (90% CI 33–44); P(Poor) = 0.81", "Deterministic formula on posterior samples of uncertain inputs"],
        ["Per-fuel savings", "Gas +6,100 kWh/yr (CI …); electricity −1,900 kWh/yr", "Twin rerun on each posterior sample, before vs after"],
        ["Carbon", "tCO<sub>2</sub>e/yr with CI", "Per-fuel savings × documented emission factors"],
        ["Value impact", "+2–5% range, indicative (historical association, not a promised increase)", "Posterior predictive of the value model, recorded band vs estimated band after"],
        ["Retrofit suitability", "High / Medium / Low", "P(suitable) = P(feasible) × P(net benefit): ≥ 0.8 High, 0.5–0.8 Medium"],
        ["Data confidence", "High / Medium / Low", "Months of meter data, EPC age, imputed fields; drives interval width"],
    ], [36 * mm, 58 * mm, 76 * mm], zebra=True)
    P("Illustrative figures only. Calibration is checked on held-out homes: 90% intervals should contain the truth about "
      "90% of the time (coverage, CRPS, PIT histogram, R-hat).", "small")

    # 7 Retrofit engine
    H1("7. Retrofit engine")
    bullets([
        "<b>Feasibility, layer 1 (rules).</b> Deterministic vetoes with reason codes: no loft, not a cavity wall, external "
        "insulation in a conservation area or on a listed building, no roof access for PV, north-facing roof, no "
        "outdoor space for a heat-pump unit, already installed. A rule veto cannot be overridden.",
        "<b>Feasibility, layer 2 (Jev).</b> For retrofits that pass the rules, Jev reads home facts plus free text "
        "(EPC descriptions, agent notes, planning details) and returns feasible / with conditions / needs survey / "
        "infeasible with a calibrated confidence; low confidence becomes \"needs survey\", never a silent pass.",
        "<b>Ordering.</b> Fabric first, then glazing, controls, heating system (sized on reduced demand), then generation "
        "and storage. Marginal savings are computed in sequence on the twin, so interactions are explicit.",
        "<b>Multi-objective Pareto.</b> All feasible packages (≤ 256) are enumerated and evaluated on net cost after "
        "grants, carbon, bill saving, score gain and value. The front is computed for every posterior sample, giving "
        "<b>P(on front)</b>; the knee point is the default suggestion and a budget slider shrinks the front live.",
    ])
    H1("8. Energy savings and carbon")
    P("Savings are estimated per retrofit (marginal, in sequence) and for the package, separately for electricity and gas, "
      "against one consistent baseline. An increase is a negative saving: a heat pump removes gas use and adds electricity. "
      "All savings are labelled estimates, not guarantees.")
    table([
        ["Input", "Value used", "Source (verified 10 Oct 2026)"],
        ["Electricity factor", "0.19553 kgCO<sub>2</sub>e/kWh = generation 0.177 + transmission and distribution 0.01853", "DESNZ GHG conversion factors 2025, flat file rows 7_400_4000_5_1 and 13_402_4000_5_1"],
        ["Gas factor", "0.18296 kgCO<sub>2</sub>e/kWh, gross calorific value (matches meter kWh)", "DESNZ 2025, row 1_100_1004_6_1"],
        ["Tariffs", "Electricity 26.32 p/kWh, gas 7.97 p/kWh (1 Oct – 31 Dec 2026)", "Ofgem price cap announcement"],
        ["Retrofit costs", "BEIS 2017 Table 3 low / mean / high per home × CPI 1.3888 (ONS D7BT, 2017 → Aug 2026)", "BEIS Domestic cost assumptions (2017); ONS CPI"],
        ["Heat pump, PV costs", "£12,500 per 8 kW heat pump; £1,876 per kW PV (up to 4 kW)", "DESNZ data as reported by Which? and pv magazine; marked secondary until confirmed"],
    ], [30 * mm, 70 * mm, 70 * mm], zebra=True)
    P("Carbon is ΔC = Σ<sub>f</sub> ΔE<sub>f</sub> × EF<sub>f</sub>. Any figure without a documented source is not shown: "
      "unsourced retrofit costs (triple and secondary glazing, storage heaters, batteries) return no cost and are left out "
      "of cost-based ranking, and a fuel without a sourced factor returns no carbon figure, each with the reason.", "small")
    # 9 Value
    H1("9. Property value impact")
    P("Value impact is always a range and always labelled an indicative estimate based on historical and market "
      "relationships in our data, not a proven causal effect.", "lead")
    bullets([
        "<b>Data.</b> The UCL attribute-linked Land Registry Price Paid and EPC dataset (millions of sales with floor area, "
        "rooms and energy information) plus the UK House Price Index for market conditions. The current valuation used "
        "as baseline is kept separate from historical sale prices.",
        "<b>Hedonic quantile model.</b> Price quantiles from location, size, type, age, period and EPC band; predict the "
        "same home before and after the proposed improvements, holding everything else constant; the band after "
        "retrofit comes from the documented rating estimator (section 4).",
        "<b>Green-twin matching.</b> Pairs of real homes that match on type, built form, age band, tenure, LSOA, sale "
        "quarter and floor area (±10%) but differ by at least 15 score points; balance checked (standardised mean "
        "difference &lt; 0.1). Measures the price and running-cost difference per 10 points, benchmarked against "
        "published estimates (e.g. Nationwide: A/B about 1.6–1.7% above comparable D homes).",
        "<b>DR-learner.</b> On repeat sales, a doubly robust causal meta-learner estimates how much an EPC upgrade changes "
        "price for different homes, with sensitivity analysis and refutation tests for unmeasured confounding (such as "
        "other renovations).",
    ])
    H1("10. Grounded LLM and what-if questions")
    bullets([
        "<b>Tool-only.</b> The LLM can call get_home_facts, get_results, run_scenario, search_docs and templated graph "
        "queries. It never computes; every what-if reruns run_scenario first.",
        "<b>Validator.</b> Every number in an answer is extracted and matched against that turn's tool outputs; a "
        "mismatch triggers regeneration, then a template fallback. A second Jev check asks whether every claim is "
        "supported by the JSON.",
        "<b>System 1 / System 2 routing.</b> Jev classifies the question (numeric, comparative, feasibility, "
        "out-of-scope) and fills the scenario with a calibrated confidence; below the threshold the LLM asks a "
        "clarifying question. Numbers always come from the physics tool (System 2). Without a Jev key, OpenAI gpt-6-luna with "
        "structured output fills the same interface.",
        "<b>Honesty.</b> Missing or conflicting data, infeasible retrofits and electricity increases are stated plainly; "
        "loan figures are illustrative; a human adviser makes credit decisions.",
    ])
    story.append(PageBreak())

    # 11 Weather & geography
    H1("11. Weather, climate and geography")
    bullets([
        "<b>Weather normalisation.</b> Per-home degree-day regression (kWh = baseload + UA<sub>eff</sub> × HDD, base 15.5 °C) "
        "separates a cold year from a poor house and annualises missing months from that month's actual degree days.",
        "<b>Sources.</b> Met Office HadUK-Grid (1 km daily temperature, Open Government Licence) and UKCP18 for documented "
        "figures; Open-Meteo for hourly, forecast and CMIP6 projections; PVGIS for solar yield. Responses are cached.",
        "<b>Trends and lifetime value.</b> Regional heating- and cooling-degree-day trends (with intervals) on the map; "
        "future-climate scenarios turn first-year savings into lifetime savings and carbon; an overheating flag for "
        "homes where insulation without shading could raise summer temperatures.",
        "<b>Geography.</b> Postcode → LSOA via the ONS directory; conservation areas, listed buildings and off-gas-grid "
        "status feed feasibility and grants; hotspot analysis finds clusters of poor homes; the spatial hierarchy lets "
        "sparse areas borrow strength in the Bayesian model.",
    ])

    # 12 Models
    H1("12. Models we build")
    table([
        ["Model", "Kind", "Purpose", "Baseline"],
        ["Green Home Score", "Deterministic", "Score, band, contributions", "—"],
        ["Post-retrofit rating estimator", "SAP 2012 ECF on twin", "Estimated rating and band after retrofit", "No change"],
        ["Level-1 twin + synthetic generator", "Physics / simulator", "Per-fuel energy, element heat loss, labels", "NEED medians"],
        ["Energy-intensity EBM", "Glass-box ML", "Expected kWh/m² per fuel; prior and residual", "XGBoost; NEED median"],
        ["Bayesian calibration", "NumPyro hierarchical", "Posterior parameters → intervals", "LightGBM quantile; conformal"],
        ["Improvement-potential ranking", "Twin + ranker check", "Rank poor homes by % reduction", "Rank by total kWh"],
        ["Retrofit relevance prior", "EBM classifier", "P(EPC would recommend a measure)", "Rules only"],
        ["Value: quantile / matching / DR-learner", "ML + causal", "Indicative value range", "Hedonic OLS"],
        ["Pareto optimiser", "Optimisation", "Front, P(on front), staging", "Greedy kWh per £"],
        ["Jev calibration", "Isotonic", "Calibrated routing / feasibility confidence", "OpenAI structured output"],
        ["Stretch (GPU)", "Physics-ML / operators / CV", "RC graph, DeepONet surrogate, GraphSAGE, spatial PINO, photo-to-twin", "Level-1 twin"],
    ], [50 * mm, 32 * mm, 56 * mm, 32 * mm], zebra=True)
    story.append(PageBreak())

    # 13 Data
    H1("13. Data")
    table([
        ["Dataset", "Use", "Licence / access"],
        ["EPC domestic certificates (England & Wales)", "Property features, ratings, recommendations, feasibility", "Free; GOV.UK One Login; bulk + API"],
        ["NEED anonymised 50k", "Real metered gas and electricity with installed measures", "Open Government Licence"],
        ["UCL Land Registry–EPC linked dataset", "Sales with energy information for value models", "UK Data Service ReShare"],
        ["HM Land Registry Price Paid; UK HPI", "Sale history; market conditions", "OGL (attribution)"],
        ["London smart meters (half-hourly)", "Load shapes, solar self-consumption", "Open (London Datastore)"],
        ["Electrification of Heat trial", "Real heat-pump efficiency priors", "UK Data Service"],
        ["English Housing Survey", "Stock distributions for the generator", "Official statistics"],
        ["HadUK-Grid, UKCP18, Open-Meteo, PVGIS", "Weather, climate, solar", "OGL / CC BY 4.0 / free"],
        ["ONS postcode directory, planning.data.gov.uk, Historic England, OSM", "Geography and planning constraints", "Open"],
        ["DESNZ GHG factors, Ofgem price cap, BEIS retrofit costs", "Carbon, bills, costs", "Official publications"],
    ], [62 * mm, 66 * mm, 42 * mm], zebra=True)
    P("No real personal or customer data is used. Splits are by property (or customer), assigned once; EPC "
      "\"potential\" fields are excluded from model inputs as leakage. Missing months and fields are handled explicitly "
      "and documented. Three synthetic showcase homes drive the demo: a 1905 solid-wall terrace in a conservation area, "
      "a 1972 cavity semi, and an all-electric third-floor flat.", "small")

    # 14 UX
    H1("14. User experience")
    bullets([
        "<b>3D twin (Three.js).</b> The customer's house outside and inside (cutaway), with a thermal view coloured by "
        "Level-1 heat loss per element. Every retrofit can be added and appears on the model: insulation layers in the "
        "loft and walls, new windows, the heat-pump unit outside and cylinder inside, panels on the best roof face, a "
        "thermostat on the wall. Infeasible items are greyed out with the reason. A live panel shows score, per-fuel kWh, "
        "carbon, bills and value with intervals; a timeline plays the staged retrofit path year by year. Element colours "
        "are labelled 'estimated from home type and EPC descriptions; only the total heat loss is calibrated to your bills', "
        "because monthly bills cannot identify losses element by element.",
        "<b>Customer app.</b> Home profile, score and contribution waterfall, retrofit plan, Pareto chart, green-loan "
        "view, \"green twins of your home\" (matched greener homes shown as ranges, never addresses), what-if chat and "
        "plan history.",
        "<b>Agent capture.</b> EPC pre-fill, floor-plan and room photos, element checklist, consent and blurring, twin on site.",
        "<b>Lloyds dashboard.</b> Heat map with switchable layers, segments, Green Profits and funnel.",
    ])
    story.append(PageBreak())

    # 15 Evaluation
    H1("15. Evaluation")
    table([
        ["Component", "Held-out metric", "Baseline"],
        ["Energy model", "MAE / MAPE per fuel; 90% interval coverage; CRPS", "NEED median by type, size, region"],
        ["Savings", "Error vs held-out reference labels and NEED before/after cases", "Sum of illustrative percentages"],
        ["Calibration", "Coverage ≈ 90%; flat PIT; R-hat &lt; 1.01", "Quantile regression; conformal"],
        ["Improvement ranking", "Spearman / NDCG", "Rank by total consumption"],
        ["Recommendations", "Precision@k vs feasibility labels; zero infeasible recommended", "Recommend everything"],
        ["Value", "Interval coverage on held-out sales; method check only: recovery of a premium we inserted into synthetic values", "Hedonic OLS"],
        ["Score", "Monotonicity sweep: 0 violations; reproducibility", "—"],
        ["LLM", "Groundedness ≥ 90%, accuracy ≥ 90%, unsupported &lt; 5% on 30–100 adversarial scenarios", "LLM without validator"],
        ["Pareto", "Hypervolume", "Greedy kWh per £"],
        ["Jev", "Routing accuracy, feasibility precision/recall, calibration error", "Rules; OpenAI structured output"],
    ], [34 * mm, 82 * mm, 54 * mm], zebra=True)
    P("Adversarial scenarios cover missing months, conflicting data (rating A but 400 kWh/m²), infeasible requests "
      "(a heat pump in a third-floor flat), electricity rising after a heat-pump switch and too little data to estimate "
      "value. Every reported metric is reproducible from a seeded script.", "small")

    H1("16. Responsible AI")
    bullets([
        "<b>Privacy.</b> Public and synthetic data only; location shown as postcode district or LSOA; interior photos need "
        "explicit consent, are blurred and kept only to build the twin.",
        "<b>Explainability.</b> Glass-box score and EBM; provenance on every number; reasons from rule codes.",
        "<b>Hallucination control.</b> Tool-only LLM, numeric validator, cached and replayable calls.",
        "<b>Bias.</b> Checks that large, older or electric homes are not penalised for size or fuel alone; per-segment metrics.",
        "<b>Human oversight.</b> Loans are illustrative; credit decisions stay with a human adviser; low-confidence feasibility "
        "becomes a survey, not an automatic answer.",
        "<b>Limits stated.</b> Synthetic labels, observational value effects and vendor model claims are labelled as such.",
    ])
    H1("17. Compliance with the brief")
    table([
        ["Constraint (brief section)", "How GreenTwin meets it"],
        ["Figures only from documented models (core rule)", "Level-1 twin, score mapping, rating estimator, value models; LLM is tool-only with a numeric validator"],
        ["Projected score and value need a rating after retrofit (2, 7)", "Documented SAP 2012 ECF estimator anchored to the recorded SAP; labelled 'estimated rating, not an EPC'"],
        ["Intensity = annualised metered energy per m² (2)", "Exactly that; weather-normalised only as a sensitivity"],
        ["Justify inputs affecting two components (2)", "Fabric/heating vs rating and PV vs intensity justified; PV overlap size reported"],
        ["Documented, monotonic sub-score mappings (2)", "Every heating system mapped (test-enforced); monotonicity tests"],
        ["Per-fuel savings, interactions, negative savings (3, 5)", "Twin reruns on packages; electricity and gas reported separately"],
        ["Documented emission factors with units and source (6)", "DESNZ 2025 rows quoted; electricity incl. T&amp;D, gas gross CV"],
        ["Retrofit catalogue source and assumptions (4)", "BEIS 2017 + ONS CPI; secondary and unsourced items flagged; no unsourced cost shown"],
        ["Value as a range, indicative, causal limits (7, 9)", "Three methods, fixed wording, never a reason to borrow; sensitivity and refutation tests"],
        ["Synthetic labels plausible, rules disclosed (9)", "Physics generator validated on NEED; synthetic premium only a method check; Green Profits is a scenario with no model trained on synthetic labels"],
        ["No real personal or customer data (9)", "Public and synthetic data only; synthetic customer base labelled"],
    ], [62 * mm, 108 * mm], zebra=True)
    story.append(PageBreak())

    # 18 Plan
    H1("18. Build plan and team")
    P("Five members with clear ownership: Data & Geography; Physics & Bayesian; ML, Value & Causal; LLM, Knowledge base, "
      "Jev & API; Front-end & 3D. We build a thin end-to-end path first, then deepen in order of rubric weight × "
      "confidence; each checkpoint leaves a submittable system.", "lead")
    table([
        ["Tier", "Contents", "Checkpoint"],
        ["A — required outputs", "Contracts and splits; Level-1 twin; score; rules and catalogue; run_scenario; synthetic generator; EBM/XGBoost; value ranges", "Score, recommendations, per-fuel savings, carbon and value for 3 homes"],
        ["B — grounded product", "Tool-only LLM + validator; 2D dashboard and chat; evaluation harness", "Complete, evaluated submission"],
        ["C — differentiators", "Weather normalisation; Bayesian calibration; Pareto; geography and Lloyds dashboard; 3D twin; Jev; matching and DR-learner; memory and agent capture", "Demo build; feature freeze"],
        ["D — stretch (GPU)", "Differentiable RC graph; DeepONet; GraphSAGE; spatial PINO; photo-to-twin; UKCP18 lifetime savings", "Only after C"],
    ], [34 * mm, 92 * mm, 44 * mm], zebra=True)
    H2("Confidence map")
    P("Each technique carries a Bayesian confidence score: a Beta prior on the chance of finishing it well (dot = mean, "
      "line ≈ 80% interval). These are planning estimates, updated at each checkpoint.", "small")
    story.append(confidence_chart())
    story.append(Spacer(1, 6))
    H2("Main risks")
    table([
        ["Risk", "Mitigation"],
        ["Too many techniques", "Tiers, checkpoints and stop rules; stretch work only after the demo build"],
        ["Synthetic labels mirror the generator", "Validate against NEED; state the limitation"],
        ["Value read as causal", "Ranges, \"indicative\", sensitivity and refutation tests"],
        ["API keys or Jev access unavailable", "Same interfaces with OpenAI, rules or cached fallbacks"],
        ["Live demo failure", "Precomputed showcase packages, offline mode, recorded video"],
    ], [60 * mm, 110 * mm])

    refs = [Paragraph("References", S["h1"])]
    for r in [
        "Megathon X: Singularity, Problem Statement 2 (Lloyds): Green Home Score and Retrofit Recommendation Engine.",
        "MHCLG, Get energy performance of buildings data (EPC). DESNZ, National Energy Efficiency Data-Framework (NEED) anonymised data.",
        "HM Land Registry Price Paid Data and UK House Price Index (OGL v3). Chi, Dennett, Oléron-Evans, Morphet, attribute-linked residential property price dataset (UK Data Service 854942 / 857911).",
        "DESNZ 2025 greenhouse gas conversion factors. Ofgem energy price cap, 1 Oct – 31 Dec 2026. BEIS, Domestic cost assumptions (2017).",
        "Met Office HadUK-Grid; UKCP18; Open-Meteo; EC JRC PVGIS. Energy Systems Catapult, Electrification of Heat demonstration.",
        "Fuerst, McAllister, Nanda, Wyatt (2013), An investigation of the effect of EPC ratings on house prices (DECC). Nationwide EPC price-premium analysis.",
        "InterpretML (EBM); NumPyro; EconML; DoWhy; pymoo; Three.js / react-three-fiber. TypeSafe Jev (System-1 model), vendor documentation.",
    ]:
        refs.append(Paragraph(r, S["bullet"], bulletText="\u2022"))
    story.append(KeepTogether(refs))


def footer(c, doc):
    c.saveState()
    c.setFont(F, 7.5)
    c.setFillColor(GREY)
    c.drawString(20 * mm, 10 * mm, "GreenTwin · Team MegaMind · Megathon X, Lloyds PS2 · M1")
    c.drawRightString(A4[0] - 20 * mm, 10 * mm, f"{doc.page}")
    c.restoreState()


def main():
    build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=16 * mm, bottomMargin=18 * mm, title="GreenTwin — M1 approach and idea",
                            author="Team MegaMind")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
