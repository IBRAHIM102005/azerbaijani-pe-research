#!/usr/bin/env python
"""Generate the paper figure showing where each PE scheme enters the model.

Writes ``report/figures/pe_injection_points.svg``.  No plotting dependency is
used, so the figure regenerates identically on any machine and in CI.

For the IEEE template, convert once:

    rsvg-convert -f pdf -o pe_injection_points.pdf pe_injection_points.svg
    # or: inkscape --export-type=pdf pe_injection_points.svg

Usage
-----
    python scripts/make_pe_diagram.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures" / "pe_injection_points.svg"

W, H = 720, 690
INK = "#1b1f23"
MUTED = "#6a737d"
FILL = "#ffffff"
ACCENT = {"A": "#2f6f4f", "B": "#8a5a00", "C": "#3b5b9e"}


def box(x, y, w, h, label, sub=None, dashed=False, stroke=INK, fill=FILL, rx=6):
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    out = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.4"{dash}/>'
    ]
    if sub is None:
        out.append(
            f'<text x="{x + w / 2}" y="{y + h / 2 + 5}" text-anchor="middle" '
            f'font-size="14" fill="{INK}">{label}</text>'
        )
    else:
        out.append(
            f'<text x="{x + w / 2}" y="{y + h / 2 - 3}" text-anchor="middle" '
            f'font-size="14" fill="{INK}">{label}</text>'
        )
        out.append(
            f'<text x="{x + w / 2}" y="{y + h / 2 + 15}" text-anchor="middle" '
            f'font-size="11.5" fill="{MUTED}">{sub}</text>'
        )
    return out


def arrow(x1, y1, x2, y2, stroke=INK, dashed=False, marker="arrow"):
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    return [
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
        f'stroke-width="1.4" marker-end="url(#{marker})"{dash}/>'
    ]


def tag(x, y, letter):
    color = ACCENT[letter]
    return [
        f'<circle cx="{x}" cy="{y}" r="11" fill="{color}"/>',
        f'<text x="{x}" y="{y + 4.5}" text-anchor="middle" font-size="12.5" '
        f'font-weight="700" fill="#ffffff">{letter}</text>',
    ]


def build() -> str:
    s: list[str] = []
    cx, bw = 100, 250          # main column left edge and width
    mid = cx + bw / 2

    s.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'width="{W}" height="{H}" font-family="Helvetica, Arial, sans-serif">'
    )
    s.append(
        '<defs>'
        '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{INK}"/></marker>'
        '<marker id="arrowA" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{ACCENT["A"]}"/></marker>'
        '<marker id="arrowB" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{ACCENT["B"]}"/></marker>'
        '<marker id="arrowC" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{ACCENT["C"]}"/></marker>'
        '</defs>'
    )
    s.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

    # ---- main column ----------------------------------------------------
    s += box(cx, 24, bw, 34, "input token ids", rx=17)
    s += arrow(mid, 58, mid, 78)

    s += box(cx, 78, bw, 44, "token embedding", "wte, untied output head")
    s += tag(cx + bw + 40, 100, "A")
    s += arrow(mid, 122, mid, 148)

    # transformer block frame
    s.append(
        f'<rect x="{cx - 26}" y="{150}" width="{bw + 52}" height="{330}" rx="10" '
        f'fill="none" stroke="{MUTED}" stroke-width="1.2" stroke-dasharray="4 4"/>'
    )
    s.append(
        f'<text x="{cx - 20}" y="{168}" font-size="11.5" fill="{MUTED}">'
        f'transformer block  \u00d7 6  (pre-LN, parallel residual)</text>'
    )

    s += box(cx, 178, bw, 34, "LayerNorm")
    s += arrow(mid, 212, mid, 230)
    s += box(cx, 230, bw, 34, "q, k, v projection")
    s += arrow(mid, 264, mid, 282)

    s += box(cx, 282, bw, 40, "rotate q and k", dashed=True, stroke=ACCENT["B"])
    s += tag(cx + bw + 40, 302, "B")
    s += arrow(mid, 322, mid, 340)

    s += box(cx, 340, bw, 40, "attention scores",
             "scaled dot product")
    s += arrow(mid, 374, mid, 392)

    s += box(cx, 392, bw, 42, "+ causal mask", "+ distance bias", dashed=True,
             stroke=ACCENT["C"])
    s += tag(cx + bw + 40, 413, "C")
    s += arrow(mid, 434, mid, 452)

    s += box(cx, 452, bw, 0, "")   # spacer removed
    s = [e for e in s if 'height="0"' not in e]
    s += box(cx, 446, bw, 30, "softmax \u2192 attn branch")

    s += arrow(mid, 480, mid, 500)
    s += box(cx, 500, bw, 34, "final LayerNorm")
    s += arrow(mid, 534, mid, 552)
    s += box(cx, 552, bw, 34, "LM head \u2192 logits", rx=17)

    # ---- callouts -------------------------------------------------------
    lx, lw = 420, 268

    s.append(
        f'<text x="{lx}" y="{40}" font-size="13" font-weight="700" fill="{INK}">'
        f'injection points</text>'
    )

    s += box(lx, 56, lw, 76, "", dashed=False, stroke=ACCENT["A"])
    s.append(
        f'<text x="{lx + 14}" y="{78}" font-size="12.5" font-weight="700" '
        f'fill="{ACCENT["A"]}">A: absolute, on the embedding</text>'
    )
    s.append(
        f'<text x="{lx + 14}" y="{97}" font-size="11.5" fill="{INK}">'
        f'learned: trainable table (512 x 256), 131,072 params</text>'
    )
    s.append(
        f'<text x="{lx + 14}" y="{114}" font-size="11.5" fill="{INK}">'
        f'sinusoidal: fixed sin/cos table, 0 params</text>'
    )
    s += arrow(lx, 94, cx + bw + 54, 100, stroke=ACCENT["A"], dashed=True,
               marker="arrowA")

    s += box(lx, 258, lw, 76, "", stroke=ACCENT["B"])
    s.append(
        f'<text x="{lx + 14}" y="{280}" font-size="12.5" font-weight="700" '
        f'fill="{ACCENT["B"]}">B: relative, inside attention</text>'
    )
    s.append(
        f'<text x="{lx + 14}" y="{299}" font-size="11.5" fill="{INK}">'
        f'RoPE: rotates q, k by angle m\u03b8; the logit</text>'
    )
    s.append(
        f'<text x="{lx + 14}" y="{316}" font-size="11.5" fill="{INK}">'
        f'then depends on (m \u2212 n) only. 0 params</text>'
    )
    s += arrow(lx, 296, cx + bw + 54, 302, stroke=ACCENT["B"], dashed=True,
               marker="arrowB")

    s += box(lx, 372, lw, 76, "", stroke=ACCENT["C"])
    s.append(
        f'<text x="{lx + 14}" y="{394}" font-size="12.5" font-weight="700" '
        f'fill="{ACCENT["C"]}">C: relative, on the logits</text>'
    )
    s.append(
        f'<text x="{lx + 14}" y="{413}" font-size="11.5" fill="{INK}">'
        f'ALiBi: adds \u2212slope<tspan font-size="9" dy="3">h</tspan>'
        f'<tspan dy="-3"> \u00b7 (m \u2212 n) per head,</tspan></text>'
    )
    s.append(
        f'<text x="{lx + 14}" y="{430}" font-size="11.5" fill="{INK}">'
        f'slope<tspan font-size="9" dy="3">h</tspan>'
        f'<tspan dy="-3"> = 2<tspan font-size="10" dy="-4">\u22128h/H</tspan>'
        f'<tspan dy="4">. 0 params</tspan></tspan></text>'
    )
    s += arrow(lx, 410, cx + bw + 54, 413, stroke=ACCENT["C"], dashed=True,
               marker="arrowC")

    s += box(lx, 470, lw, 60, "", stroke=MUTED, dashed=True)
    s.append(
        f'<text x="{lx + 14}" y="{492}" font-size="12.5" font-weight="700" '
        f'fill="{MUTED}">NoPE: none of A, B, C</text>'
    )
    s.append(
        f'<text x="{lx + 14}" y="{511}" font-size="11.5" fill="{INK}">'
        f'position is visible only through causal masking.</text>'
    )

    s.append(
        f'<text x="{lx}" y="{560}" font-size="11" fill="{MUTED}">'
        f'Everything outside the dashed boxes is byte-identical</text>'
    )
    s.append(
        f'<text x="{lx}" y="{576}" font-size="11" fill="{MUTED}">'
        f'across the five arms, including initial weights.</text>'
    )

    s.append(
        f'<text x="{cx - 26}" y="{H - 22}" font-size="11.5" fill="{MUTED}">'
        f'Fig. 1. Where each positional encoding scheme enters the shared base '
        f'model.</text>'
    )
    s.append("</svg>")
    return "\n".join(s)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build(), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
