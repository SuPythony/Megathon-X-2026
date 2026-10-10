"""Build LLD.pdf (repo root): GreenTwin low-level design.

Hand-written sections: deliverables/src/lld_content.md and deliverables/src/lld_data.md.
Generated sections (always in sync with the code): contracts by part (contracts/models.py) and
interfaces (greentwin/interfaces.py). Rendered Markdown -> HTML -> PDF with headless Chrome.

Run:  uv run --no-project --with 'pydantic>=2.7' --with numpy --with scipy --with markdown \
          python deliverables/src/build_lld.py
"""

from __future__ import annotations

import datetime as dt
import enum
import html
import inspect
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import markdown  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from pydantic.json_schema import models_json_schema  # noqa: E402

from contracts import models  # noqa: E402
from contracts.export_ts import _ts  # noqa: E402
from greentwin import interfaces  # noqa: E402

OUT_PDF = ROOT / "LLD.pdf"
OUT_MD = SRC / "LLD.generated.md"


def _cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


# --------------------------------------------------------------------------- generated: contracts
def _sections() -> list[tuple[int, str]]:
    src = Path(models.__file__).read_text(encoding="utf-8").splitlines()
    marks = [(1, "Shared primitives")]
    for i, line in enumerate(src, 1):
        m = re.match(r"# (Part \d.*|Retrofit catalogue.*|Orchestration.*)$", line)
        if m:
            marks.append((i, m.group(1)))
    return marks


def contracts_md() -> str:
    classes = [v for v in vars(models).values() if isinstance(v, type) and v.__module__ == models.__name__
               and (issubclass(v, BaseModel) or issubclass(v, enum.Enum)) and v is not BaseModel]
    pyd = [c for c in classes if issubclass(c, BaseModel)]
    _, top = models_json_schema([(m, "validation") for m in pyd], ref_template="#/$defs/{model}")
    defs = top["$defs"]
    marks = _sections()

    def part_of(cls) -> str:
        line = inspect.getsourcelines(cls)[1]
        name = marks[0][1]
        for ln, title in marks:
            if ln <= line:
                name = title
        return name

    by_part: dict[str, list] = {}
    for c in sorted(classes, key=lambda c: inspect.getsourcelines(c)[1]):
        by_part.setdefault(part_of(c), []).append(c)

    out = ["## 5. Contracts reference (generated from `contracts/models.py`)", "",
           f"Schema version **{models.SCHEMA_VERSION}**. Wire format = JSON with these exact field names; "
           "TypeScript mirror = `contracts/types.ts` (generated). `?` = optional (may be omitted or null).", ""]
    for part, cls_list in by_part.items():
        out += [f"### {part}", ""]
        enums = [c for c in cls_list if issubclass(c, enum.Enum)]
        if enums:
            out += ["| Enum | Values |", "| --- | --- |"]
            out += [f"| `{e.__name__}` | {', '.join('`' + m.value + '`' for m in e)} |" for e in enums]
            out.append("")
        for c in cls_list:
            if not issubclass(c, BaseModel):
                continue
            d = defs.get(c.__name__, {})
            out.append(f"#### `{c.__name__}`")
            if c.__doc__:
                out += ["", _cell(inspect.cleandoc(c.__doc__))]
            out += ["", "| Field | Type | Description |", "| --- | --- | --- |"]
            req = set(d.get("required", []))
            for prop, ps in d.get("properties", {}).items():
                opt = "" if prop in req else "?"
                out.append(f"| `{prop}{opt}` | `{_cell(_ts(ps))}` | {_cell(ps.get('description', ''))} |")
            out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------- generated: interfaces
PROTOCOLS = ["DataLayer", "DigitalTwin", "EnergyRetrofitEngine", "GreenHomeScorer", "ValueModel",
             "FeasibilityEngine", "Router", "Explainer", "Orchestrator"]
PART = {"DataLayer": "Part 0", "DigitalTwin": "Part 1", "EnergyRetrofitEngine": "Part 2",
        "GreenHomeScorer": "Part 3", "ValueModel": "Part 4", "FeasibilityEngine": "Part 5",
        "Router": "Part 5", "Explainer": "Part 5", "Orchestrator": "Orchestration"}


def _sig(fn) -> str:
    sig = inspect.signature(fn)
    parts = []
    for name, p in sig.parameters.items():
        if name == "self":
            continue
        s = name
        if p.annotation is not inspect.Parameter.empty:
            s += f": {p.annotation}"
        if p.default is not inspect.Parameter.empty:
            s += f" = {p.default!r}"
        parts.append(s)
    ret = sig.return_annotation
    return f"({', '.join(parts)})" + (f" -> {ret}" if ret is not inspect.Signature.empty else "")


def interfaces_md() -> str:
    out = ["## 6. Interfaces reference (generated from `greentwin/interfaces.py`)", "",
           "Each part implements its Protocol; callers depend only on the Protocol (stub first, real later). "
           "`PosteriorSamples` and `TwinRun` are in-process dataclasses holding arrays over posterior samples.", "",
           "```python", inspect.getsource(interfaces.PosteriorSamples).rstrip(), "",
           inspect.getsource(interfaces.TwinRun).rstrip(), "```", ""]
    for name in PROTOCOLS:
        proto = getattr(interfaces, name)
        out += [f"### `{name}` ({PART[name]})", ""]
        if proto.__doc__:
            out += [_cell(inspect.cleandoc(proto.__doc__)), ""]
        out += ["| Method | Signature | Contract / behaviour |", "| --- | --- | --- |"]
        for mname, fn in proto.__dict__.items():
            if mname.startswith("_") or not callable(fn):
                continue
            doc = inspect.cleandoc(fn.__doc__) if fn.__doc__ else ""
            out.append(f"| `{mname}` | `{_cell(_sig(fn))}` | {_cell(doc)} |")
        out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------- render
CSS = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm; }
body { font-family: 'DejaVu Sans', 'Segoe UI', Arial, sans-serif; font-size: 9.2pt; line-height: 1.42; color: #1d1d1d; }
h1 { color: #0b6e4f; font-size: 22pt; margin: 0 0 4pt 0; }
h2 { color: #0b6e4f; font-size: 14.5pt; border-bottom: 2px solid #0b6e4f; padding-bottom: 3pt; margin-top: 18pt; break-before: page; }
h2.first { break-before: auto; }
h3 { font-size: 11.5pt; margin: 14pt 0 4pt 0; color: #1d1d1d; }
h4 { font-size: 10pt; margin: 10pt 0 2pt 0; color: #2a6fdb; break-after: avoid; }
p { margin: 3pt 0 6pt 0; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 10pt 0; font-size: 8.1pt; break-inside: auto; }
th { background: #e6f2ec; text-align: left; }
th, td { border: 0.5pt solid #c9c9c9; padding: 2.5pt 4pt; vertical-align: top; }
tr { break-inside: avoid; }
tr:nth-child(even) td { background: #f8f8f8; }
code { font-family: 'DejaVu Sans Mono', Consolas, monospace; font-size: 7.9pt; background: #f2f2ee; padding: 0 2pt; border-radius: 2pt; }
pre { background: #f4f4f0; border-left: 3px solid #0b6e4f; padding: 6pt 8pt; font-size: 7.6pt; line-height: 1.35;
      white-space: pre-wrap; break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 7.6pt; }
td code { font-size: 7.4pt; background: none; padding: 0; }
hr { border: none; border-top: 1px solid #c9c9c9; margin: 10pt 0; }
ul, ol { margin: 2pt 0 6pt 0; padding-left: 16pt; }
.toc { columns: 2; font-size: 9pt; }
"""


def _esc(text: str) -> str:
    """Stop Markdown from reading field names like CURRENT_/POTENTIAL_ as emphasis."""
    return text.replace("_", r"\_").replace("*", r"\*")


def data_md() -> str:
    sys.path.insert(0, str(SRC))
    from datasets import DATASETS, DAY1, VERIFIED_DATE  # noqa: E402

    table = ["| Dataset | Used by | Access | Auth | Licence |", "| --- | --- | --- | --- | --- |"]
    table += [f"| **{_cell(x['name'])}** | {_cell(x['use'])} | {_cell(x['access'])} | {_cell(x['auth'])} | "
              f"{_cell(x['licence'])} |" for x in DATASETS]
    detail = []
    for x in DATASETS:
        detail += [f"#### {x['name']}", "",
                   f"- **Publisher / page:** {x['publisher']}: <{x['url']}>",
                   f"- **Format / size / coverage:** {x['format']}; {x['coverage']}",
                   f"- **Key fields:** {_esc(x['fields'])}",
                   f"- **Gotchas:** {_esc(x['gotchas'])}"]
        if x.get("how"):
            detail += ["", "```bash", x["how"].strip(), "```"]
        detail.append("")
    day1 = "\n".join(f"{i}. {s}" for i, s in enumerate(DAY1, 1))
    data = (SRC / "lld_data.md").read_text(encoding="utf-8")
    return (data.replace("{{VERIFIED_DATE}}", VERIFIED_DATE).replace("{{DATASETS_TABLE}}", "\n".join(table))
                .replace("{{DAY1}}", day1).replace("{{DATASETS_DETAIL}}", "\n".join(detail)))


def build_markdown() -> str:
    content = (SRC / "lld_content.md").read_text(encoding="utf-8")
    data = data_md()
    return (content.replace("{{SCHEMA_VERSION}}", models.SCHEMA_VERSION)
                   .replace("{{BUILD_DATE}}", dt.date.today().isoformat())
                   .replace("{{CONTRACTS}}", contracts_md())
                   .replace("{{INTERFACES}}", interfaces_md())
                   .replace("{{DATA}}", data))


def toc(md: str) -> str:
    items = [re.sub(r"`", "", m) for m in re.findall(r"^## (.+)$", md, flags=re.M)]
    return "<div class='toc'><b>Contents</b><ol style='list-style:none;padding-left:0'>" + "".join(
        f"<li>{html.escape(i)}</li>" for i in items) + "</ol></div>"


def render_pdf(md: str) -> None:
    body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
    body = body.replace("<h2>", "<h2 class='first'>", 1)
    first_hr = body.find("<hr />")
    if first_hr != -1:
        body = body[:first_hr] + toc(md) + body[first_hr:]
    page = f"<!doctype html><html><head><meta charset='utf-8'><title>GreenTwin LLD</title><style>{CSS}</style></head>" \
           f"<body>{body}</body></html>"
    chrome = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")
    if not chrome:
        raise SystemExit("needs google-chrome/chromium for HTML -> PDF")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "lld.html"
        src.write_text(page, encoding="utf-8")
        subprocess.run([chrome, "--headless=new", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={OUT_PDF}", src.as_uri()], check=True, capture_output=True, timeout=180)


def main() -> None:
    md = build_markdown()
    OUT_MD.write_text(md, encoding="utf-8")
    render_pdf(md)
    print("wrote", OUT_PDF, "and", OUT_MD)


if __name__ == "__main__":
    main()
