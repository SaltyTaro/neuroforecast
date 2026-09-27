"""Render docs/technical_report.md to a print-ready PDF (docs/technical_report.pdf) with headless Chrome.

Needs markdown-it-py and playwright (using the installed Chrome channel). Figures are embedded from docs/figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]

CSS = """
@page { size: A4; margin: 18mm 17mm 18mm 17mm; }
:root { --ink: #0b0b0b; --ink2: #52514e; --rule: #dcdbd6; --tint: #f4f3ef; --accent: #2a78d6; }
html { font-family: "Segoe UI", "Microsoft YaHei", "Noto Sans CJK SC", Arial, sans-serif; font-size: 10.2pt; color: var(--ink); }
body { line-height: 1.45; }
h1 { font-size: 17pt; line-height: 1.2; margin: 0 0 6pt; }
h2 { font-size: 12.5pt; margin: 16pt 0 5pt; padding-top: 4pt; border-top: 1px solid var(--rule); break-after: avoid; }
h3 { font-size: 10.8pt; margin: 11pt 0 4pt; break-after: avoid; }
p, li { orphans: 3; widows: 3; }
p { margin: 0 0 6pt; }
ul, ol { margin: 0 0 6pt 16pt; padding: 0; }
a { color: var(--accent); text-decoration: none; }
code { font-family: Consolas, "Cascadia Mono", monospace; font-size: 8.8pt; background: var(--tint); padding: 0 2px; border-radius: 3px; }
pre { background: var(--tint); padding: 6pt 8pt; border-radius: 4px; font-size: 8.2pt; line-height: 1.35; overflow: hidden; white-space: pre-wrap; break-inside: avoid; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 9pt; font-size: 8.8pt; break-inside: avoid; font-variant-numeric: tabular-nums; }
th, td { border-bottom: 1px solid var(--rule); padding: 3pt 5pt; text-align: left; vertical-align: top; }
th { color: var(--ink2); font-weight: 600; border-bottom: 1.5px solid #b9b8b1; }
img { max-width: 100%; display: block; margin: 6pt auto 10pt; break-inside: avoid; }
hr { border: none; border-top: 1px solid var(--rule); margin: 10pt 0; }
strong { font-weight: 650; }
"""


def render(markdown_path: Path) -> str:
    md = MarkdownIt("commonmark", {"html": False}).enable("table")
    body = md.render(markdown_path.read_text(encoding="utf-8"))
    base = markdown_path.parent.resolve().as_uri() + "/"
    return f"<!doctype html><html lang='en'><head><meta charset='utf-8'><base href='{base}'><style>{CSS}</style></head><body>{body}</body></html>"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "docs/technical_report.md")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/technical_report.pdf")
    args = parser.parse_args()
    html_path = args.output.with_suffix(".print.html")
    html_path.write_text(render(args.input), encoding="utf-8")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page()
        page.goto(html_path.resolve().as_uri(), wait_until="networkidle")
        page.pdf(path=str(args.output), format="A4", print_background=True, prefer_css_page_size=True,
                 display_header_footer=True, header_template="<span></span>",
                 footer_template="<div style='font-size:7pt;color:#8d8c86;width:100%;text-align:center'>NeuroForecast technical report · <span class='pageNumber'></span> / <span class='totalPages'></span></div>",
                 margin={"top": "18mm", "bottom": "18mm", "left": "17mm", "right": "17mm"})
        browser.close()
    html_path.unlink()
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
