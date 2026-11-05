"""Build the README figures: python docs/assets/src/figures.py [--only NAME ...]

Writes <name>-dark.svg and <name>-light.svg into docs/assets/, plus theme-neutral
glyph-*.svg icons. Spectra and model summaries come from the tracked phase-2 runs
under analysis_runs/; nothing in a data panel is invented.
"""
from __future__ import annotations

import argparse
import math
import re
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from svgkit import Svg, fmt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "docs" / "assets"
RUNS = ROOT / "analysis_runs" / "phase2"
PYTHIA = RUNS / "family_batch_001" / "metrics" / "EleutherAI--pythia-160m.h5"
ANATOMY_LAYER = "gpt_neox.layers.5.mlp.dense_h_to_4h"
EVALS_THRESH = 1e-5


# data ----------------------------------------------------------------------
@lru_cache(maxsize=None)
def spectra(path: Path = PYTHIA):
    """Retained eigenvalues and fit results for every measured matrix."""
    with h5py.File(path) as f:
        layers = f["layers"]
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in layers["longname"][:]]
        rows = []
        for i, name in enumerate(names):
            eigs = np.asarray(f["eigs"][i], dtype=float)
            rows.append(dict(
                name=name, eigs=np.sort(eigs[eigs > EVALS_THRESH]),
                M=int(layers["M"][i]), N=int(layers["N"][i]),
                alpha=float(layers["alpha"][i]), D=float(layers["D"][i]),
                xmin=float(layers["fit_xmin"][i]), n_tail=int(layers["n_tail"][i]),
                spectral_norm=float(layers["spectral_norm"][i]), norm=float(layers["norm"][i]),
                stable_rank=float(layers["stable_rank"][i]), entropy=float(layers["entropy"][i]),
            ))
        model = f.attrs["full_name"]
        model = model.decode() if isinstance(model, bytes) else str(model)
    return model, rows


def kde(values: np.ndarray, grid: np.ndarray, bw: float) -> np.ndarray:
    z = (grid[:, None] - values[None, :]) / bw
    return np.exp(-0.5 * z * z).sum(axis=1) / (len(values) * bw * math.sqrt(2 * math.pi))


def marchenko_pastur_log10(u: np.ndarray, M: int, N: int, frob: float) -> np.ndarray:
    """Density over log10(lambda) of the squared singular values of an i.i.d.
    M x N matrix with the same Frobenius norm (entry variance frob / (M N))."""
    big, small = max(M, N), min(M, N)
    var, y = frob / (M * N), small / big
    lam = 10.0 ** u
    x = lam / big
    a, b = var * (1 - math.sqrt(y)) ** 2, var * (1 + math.sqrt(y)) ** 2
    inside = (x > a) & (x < b)
    rho = np.zeros_like(x)
    rho[inside] = np.sqrt((b - x[inside]) * (x[inside] - a)) / (2 * math.pi * var * y * x[inside])
    return rho / big * lam * math.log(10), (a * big, b * big)


@lru_cache(maxsize=None)
def catalog() -> pd.DataFrame:
    c = pd.read_csv(RUNS / "catalog.csv", low_memory=False)
    c = c[c.outcome == "success"].drop_duplicates("model_id", keep="last").copy()
    c["arch"] = c.model_class.fillna("").str.split(".").str[-1]
    c["parent"] = c.source_model.fillna("").str.split("@").str[0]
    return c


FAMILIES = [  # label, architecture pattern, color
    ("Qwen2", r"^Qwen2", "gold"),
    ("Llama", r"^Llama", "teal"),
    ("BERT", r"Bert|Roberta", "ember"),
    ("GPT-2", r"^GPT2", "blue"),
    ("GPT-Neo(X)", r"^GPTNeo", "green"),
    ("Phi · Mistral", r"^Phi|^Mistral", "sage"),
]


def family(arch: str):
    for label, pattern, color in FAMILIES:
        if re.search(pattern, arch):
            return label, color
    return "Other", "ink3"


def alpha_color(alpha: float) -> str:
    return "ember" if alpha < 2 else "gold" if alpha < 4 else "violet"


def poly(points, precision=1) -> str:
    out, last = [], None
    for x, y in points:
        p = f"{round(x, precision):g} {round(y, precision):g}"
        if p != last:
            out.append(p)
            last = p
    return "M" + "L".join(out)


def simplify(points, tol=0.12):
    """Drop interior points that lie on the line through their neighbours."""
    if len(points) < 3:
        return list(points)
    keep = [points[0]]
    for cur, nxt in zip(points[1:], points[2:]):
        (x0, y0), (x1, y1), (x2, y2) = keep[-1], cur, nxt
        if x2 == x0:
            continue
        if abs(y0 + (y2 - y0) * (x1 - x0) / (x2 - x0) - y1) > tol:
            keep.append(cur)
    keep.append(points[-1])
    return keep


def rng(seed):
    return np.random.default_rng(seed)


# shared pieces -------------------------------------------------------------
def sparkle(s: Svg, x, y, r, color="star", opacity=1.0, cls=None, attrs=""):
    """A four-pointed star."""
    k = r * 0.16
    d = (f"M{fmt(x)} {fmt(y - r)}Q{fmt(x + k)} {fmt(y - k)} {fmt(x + r)} {fmt(y)}"
         f"Q{fmt(x + k)} {fmt(y + k)} {fmt(x)} {fmt(y + r)}Q{fmt(x - k)} {fmt(y + k)} {fmt(x - r)} {fmt(y)}"
         f"Q{fmt(x - k)} {fmt(y - k)} {fmt(x)} {fmt(y - r)}Z")
    s.path(d, fill=color, opacity=opacity, cls=cls, attrs=attrs)


def stars(s: Svg, n=70, seed=0, box=None, bright=2):
    """A seeded starfield; some stars twinkle, a few are four-pointed."""
    x0, y0, x1, y1 = box or (8, 8, s.w - 8, s.h - 8)
    g = rng(seed + s.w * 7 + s.h)
    dark = s.theme == "dark"
    for k in range(n):
        x, y = x0 + g.random() * (x1 - x0), y0 + g.random() * (y1 - y0)
        r = 0.5 + g.random() ** 3 * (1.3 if dark else 1.0)
        op = (0.25 + 0.6 * g.random()) * (1 if dark else 0.45)
        tw = k % 5 == 0
        s.circle(x, y, r, fill="star", opacity=f"{op:.2f}", cls="tw" if tw else None,
                 attrs=f'style="--t:{-g.random() * 4:.2f}s;--d:{2.4 + g.random() * 3:.1f}s"' if tw else "")
    for k in range(bright):
        x, y = x0 + g.random() * (x1 - x0), y0 + g.random() * (y1 - y0)
        sparkle(s, x, y, 4 + g.random() * 4, "star", 0.9 if dark else 0.5, cls="tw",
                attrs=f'style="--t:{-g.random() * 5:.2f}s;--d:{3 + g.random() * 3:.1f}s"')


def nebula(s: Svg, blobs=None, scale=1.0):
    """Soft colored clouds that drift slowly."""
    W, H = s.w, s.h
    blobs = blobs or [(0.1, 0.1, 0.4, "neb1", 0.16), (0.92, 0.95, 0.36, "neb2", 0.13), (0.62, -0.08, 0.3, "neb3", 0.14), (0.4, 1.1, 0.22, "neb4", 0.07)]
    dark = s.theme == "dark"
    for k, (fx, fy, fr, col, op) in enumerate(blobs):
        o = op * scale * (1 if dark else 0.75)
        s.circle(fx * W, fy * H, fr * W, fill=s.radial(f"neb-{col}-{o:.3f}", col, round(o, 3)), cls="drift",
                 attrs=f'style="--d:{16 + 5 * k}s;--t:{-3 * k}s"')


def frame(s: Svg, rx=16, sky=True, n_stars=60):
    """The figure's night sky (dark) or vellum (light), clipped to rounded corners."""
    W, H = s.w, s.h
    s.define(f"frameclip", f'<clipPath id="frameclip"><rect x="1" y="1" width="{W - 2}" height="{H - 2}" rx="{rx - 1}"/></clipPath>')
    s.rect(0.5, 0.5, W - 1, H - 1, rx=rx, fill="panel")
    s.open('clip-path="url(#frameclip)"')
    if sky:
        nebula(s)
    if n_stars:
        stars(s, n_stars)
    s.close()
    rim = s.linear("rim", [(0, "teal", 0.6), (0.4, "line", 1), (0.7, "line", 1), (1, "gold", 0.6)], 0, 0, 1, 1)
    s.rect(0.5, 0.5, W - 1, H - 1, rx=rx, fill="none", stroke=rim)


def card(s: Svg, x, y, w, h, fill="panel2", stroke="line", rx=12, cls=None, attrs=""):
    if fill == "panel2" and not cls:
        attrs = f'fill-opacity="{0.72 if s.theme == "dark" else 0.86}" {attrs}'.strip()
    s.rect(x, y, w, h, rx=rx, fill=fill, stroke=stroke, cls=cls, attrs=attrs)


def chip(s: Svg, x, y, text, role="mono", size=14, fill="lift", ink="ink2", stroke="line", pad=10, h=None, anchor="start"):
    w = Svg.width(text, role, size) + 2 * pad
    h = h or size + 14
    if anchor == "middle":
        x -= w / 2
    elif anchor == "end":
        x -= w
    s.rect(x, y, w, h, rx=h / 2 if h <= 30 else 8, fill=fill, stroke=stroke)
    s.text(x + pad, y + h / 2 + size * 0.36, text, role=role, size=size, fill=ink)
    return w


ROMAN = {"01": "I", "02": "II", "03": "III", "04": "IV", "05": "V"}
STAGE = {"01": "gold_t", "02": "ember_t", "03": "teal_t", "04": "blue_t", "05": "green_t"}


def kicker(s: Svg, x, y, number, label, color=None, ink="ink3"):
    s.text(x, y, [(ROMAN.get(number, number), {"role": "serifi", "fill": color or STAGE.get(number, "gold_t"), "size": 20}), ("   " + label, {})],
           role="texti", size=16, fill=ink, ls="0.2")


def arrow(s: Svg, x1, y1, x2, y2, stroke="line2", sw=1.4, head=7, dash=None):
    s.line(x1, y1, x2, y2, stroke=stroke, sw=sw, dash=dash)
    s.arrowhead(x2, y2, math.degrees(math.atan2(y2 - y1, x2 - x1)), head, fill=stroke)


def heat_tile(s: Svg, x, y, cols, rows, cell, seed, gap=2, rank=2, cls=None):
    """A deterministic low-rank-plus-noise matrix drawn as cells (shape only, not data)."""
    g = rng(seed)
    m = sum(np.outer(g.normal(size=rows), g.normal(size=cols)) * (1.2 / (k + 1)) for k in range(rank))
    m = m + 0.6 * g.normal(size=(rows, cols))
    m = m / np.abs(m).max()
    for r in range(rows):
        for c in range(cols):
            v = m[r, c]
            col = "gold" if v > 0 else ("teal" if (r + c) % 3 else "blue")
            s.rect(x + c * cell, y + r * cell, cell - gap, cell - gap, rx=2, fill=col,
                   opacity=f"{0.12 + 0.88 * abs(v):.2f}", cls=cls)


@lru_cache(maxsize=None)
def icon(name: str):
    raw = (HERE / "icons" / f"{name}.svg").read_text()
    view = re.search(r'viewBox="([^"]+)"', raw).group(1)
    inner = re.sub(r"^.*?<svg[^>]*>|</svg>\s*$", "", raw, flags=re.S)
    inner = re.sub(r"\s+", " ", inner)
    return view, inner


def place_icon(s: Svg, name, x, y, h):
    """Black wordmark fills follow the theme ink, as brands' dark-background versions do."""
    view, inner = icon(name)
    inner = inner.replace('fill="#000000"', f'fill="{s.col("ink")}"')
    _, _, vw, vh = map(float, view.split())
    w = h * vw / vh
    uid = f"icon-{name}"
    s.define(uid, f'<symbol id="{uid}" viewBox="{view}">{inner}</symbol>')
    s.add(f'<use href="#{uid}" x="{fmt(x)}" y="{fmt(y)}" width="{fmt(w)}" height="{fmt(h)}"/>')
    return w


def pulse_ring(s: Svg, color="gold"):
    """Card outlines that light up in turn: class pulse-k with a stagger."""
    s.style(".lit{opacity:0;animation:lit var(--d,4s) ease-in-out infinite;animation-delay:var(--t,0s)}"
            "@keyframes lit{0%,100%{opacity:0}12%{opacity:1}34%{opacity:0}}")


# hero ----------------------------------------------------------------------
def mix(a: str, b: str, t: float) -> str:
    a, b = [int(a[i:i + 2], 16) for i in (1, 3, 5)], [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


def aurora(s: Svg, t: float) -> str:
    """Blue, then sea green, then green, along t in [0, 1]."""
    v, b, c = s.c["blue"], s.c["teal"], s.c["green"]
    return mix(v, b, t * 2) if t < 0.5 else mix(b, c, (t - 0.5) * 2)


def hero(theme: str) -> Svg:
    model, rows = spectra()
    W, H = 1200, 560
    s = Svg(W, H, theme, "Model-Zoo",
            f"Model-Zoo measures the eigenvalue spectrum of every weight matrix. The figure stacks the "
            f"{len(rows)} measured spectra of {model}, one ridge per matrix from the input embedding at the "
            f"back to the output embedding at the front, on a log scale. Each ridge's tail, from its fitted "
            f"cutoff, is colored by its fitted power-law exponent alpha.")
    c = s.c
    dark = theme == "dark"
    frame(s, rx=18, sky=False, n_stars=0)
    s.open('clip-path="url(#frameclip)"')
    nebula(s, [(0.06, 0.08, 0.34, "neb3", 0.14), (0.3, 1.08, 0.3, "neb2", 0.10)])
    stars(s, 110, seed=5, bright=0)
    s.close()

    X0, X1 = -2.7, 3.3
    px0, px1 = 572, 1150
    top, spacing, amp = 96, 6.9, 62
    grid = np.linspace(X0, X1, 200)
    sx = lambda v: px0 + (v - X0) / (X1 - X0) * (px1 - px0)
    s.define("ridgemask", (
        f'<linearGradient id="ridgefade" x1="{px0}" x2="{px1}" y1="0" y2="0" gradientUnits="userSpaceOnUse">'
        '<stop offset="0" stop-color="#fff" stop-opacity="0"/><stop offset=".1" stop-color="#fff"/>'
        '<stop offset=".86" stop-color="#fff"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></linearGradient>'
        f'<mask id="ridgemask" maskUnits="userSpaceOnUse" x="{px0}" y="0" width="{px1 - px0}" height="{H}">'
        f'<rect x="{px0}" y="0" width="{px1 - px0}" height="{H}" fill="url(#ridgefade)"/></mask>'))
    s.style(
        ".ridge{animation:fade 1.1s ease-out both;animation-delay:var(--t,0s)}"
        ".tail{animation:breathe 4.8s ease-in-out infinite;animation-delay:var(--t,0s)}"
        ".scan{opacity:0;animation:scan 11s cubic-bezier(.45,0,.55,1) infinite;animation-delay:var(--t,0s)}"
        f"@keyframes scan{{0%{{opacity:0;transform:translateX(0)}}10%{{opacity:1}}60%{{opacity:1}}"
        f"72%,100%{{opacity:0;transform:translateX({px1 - px0 - 120}px)}}}}")
    glow = f' filter="{s.glow_filter("tailglow", 2.4, 1.4)}"' if dark else ""
    s.open('mask="url(#ridgemask)"')
    for i, r in enumerate(rows):
        logs = np.log10(r["eigs"])
        dens = kde(logs, grid, 0.07)
        dens = dens / dens.max()
        base = top + i * spacing
        pts = [(sx(g), base - amp * d) for g, d in zip(grid, dens)]
        s.add(f'<path d="{poly(simplify(pts))}Z" fill="{c["panel"]}" stroke="{aurora(s, i / (len(rows) - 1))}" '
              f'stroke-opacity="{0.78 if dark else 0.62}" stroke-width="1.1" stroke-linejoin="round" '
              f'class="ridge" style="--t:{0.03 * i:.2f}s"/>')
        lo = math.log10(r["xmin"]) if r["xmin"] > 0 else X1
        tail = [(x, y) for (x, y), g in zip(pts, grid) if lo <= g <= logs.max() + 0.25]
        if len(tail) > 1:
            s.path(poly(simplify(tail)), stroke=alpha_color(r["alpha"]), sw=1.7, opacity=0.95,
                   cls="tail", attrs=f'style="--t:{-4.8 + 0.06 * i:.2f}s"{glow}')
    s.close()
    last = top + (len(rows) - 1) * spacing
    band = s.linear("scanband", [(0, "gold", 0), (0.5, "gold", 0.2 if dark else 0.14), (1, "gold", 0)])
    s.rect(px0 + 10, top - 70, 120, last - top + 78, fill=band, cls="scan", attrs='style="--t:2.4s"')
    s.open('clip-path="url(#frameclip)"')
    nebula(s, [(0.82, 0.1, 0.3, "neb1", 0.16), (1.0, 0.8, 0.26, "neb2", 0.14), (0.62, 0.98, 0.22, "neb3", 0.12), (0.95, 0.3, 0.14, "neb4", 0.08)])
    s.close()
    for k, (x, y, r, col) in enumerate(((1016, 58, 9, "gold"), (742, 112, 6, "star"), (1104, 176, 5, "teal"),
                                        (880, 46, 4.5, "star"), (650, 70, 5, "ember"))):
        sparkle(s, x, y, r, col, 0.95, cls="tw", attrs=f'style="--t:{-1.3 * k:.1f}s;--d:{3.2 + 0.7 * k:.1f}s"')

    for row, label in ((0, "input"), (len(rows) // 2, "depth"), (len(rows) - 1, "output")):
        s.text(px0 - 6, top + row * spacing + 5, label, role="texti", size=15, fill="ink3", anchor="end")
    s.line(px0 - 30, top + 14, px0 - 30, top + (len(rows) // 2) * spacing - 14, stroke="line2")
    s.line(px0 - 30, top + (len(rows) // 2) * spacing + 12, px0 - 30, last - 14, stroke="line2")
    axis_y = last + 26
    s.line(px0 + 40, axis_y, px1 - 40, axis_y, stroke="line2")
    for v in range(-2, 4):
        x = sx(v)
        s.line(x, axis_y, x, axis_y + 6, stroke="line2")
        s.text(x, axis_y + 24, str(v).replace("-", "−"), role="mathr", size=16, fill="ink3", anchor="middle")
    s.text(px1 - 28, axis_y + 5, [("log", {"role": "mathr"}), ("10", {"sub": True, "role": "mathr"}), (" λ", {})],
           role="math", size=19, fill="ink2")
    s.text(px0 + 40, axis_y + 52, [(model, {"role": "mono", "size": 13.5}), ("   ·   fifty measured matrices", {})],
           role="texti", size=15, fill="ink3")

    x0 = 64
    s.text(x0, 112, "Weight spectra · fingerprints · lineage", role="serifi", size=23, fill="teal_t", ls="0.5")
    wm_w = Svg.width("Model-Zoo", "serif", 112)
    wm = s.linear("wordmark", [(0, "ink", 1), (0.5, "gold" if dark else "teal_t", 1), (1, "ember" if dark else "ember_t", 1)],
                  x0, 0, x0 + wm_w, 0, units="userSpaceOnUse")
    wglow = f'filter="{s.glow_filter("wmglow", 10, 0.45)}"' if dark else ""
    s.text(x0 - 4, 224, "Model-Zoo", role="serif", size=112, fill=wm, ls="-1", attrs=wglow)
    for k, line in enumerate(["Measures the eigenvalue spectrum of every",
                              "weight matrix in a Hugging Face checkpoint,",
                              "fingerprints the same weights and publishes",
                              "the results as one queryable model zoo."]):
        s.text(x0, 284 + k * 31, line, role="sans", size=21, fill="ink2")
    eq_y, eq = 442, 25
    formulas = [
        ([("λ", {}), ("i", {"sub": True}), (" = ", {"role": "mathr"}), ("σ", {}), ("i", {"sub": True}),
          ("(", {"role": "mathr"}), ("W", {}), (")", {"role": "mathr"}), ("2", {"sup": True, "role": "mathr"})], "teal_t"),
        ([("p", {}), ("(", {"role": "mathr"}), ("λ", {}), (") ", {"role": "mathr"}), ("∝", {"role": "op"}),
          (" λ", {}), ("−α", {"sup": True})], "gold_t"),
        ([("S", {}), (" = ", {"role": "mathr"}), ("L B R", {}), ("⊤", {"sup": True, "role": "op"})], "blue_t"),
    ]
    fx = x0
    for runs, col in formulas:
        s.text(fx, eq_y, runs, role="math", size=eq, fill=col)
        fx += Svg.width(runs, "math", eq) + 34
    ly = 502
    s.text(x0, ly, "Tail color is the fitted α", role="texti", size=16, fill="ink3")
    lx = x0 + Svg.width("Tail color is the fitted α", "texti", 16) + 22
    for label, col in (("α < 2", "ember"), ("2 ≤ α < 4", "gold"), ("α ≥ 4", "violet")):
        s.line(lx, ly - 5, lx + 20, ly - 5, stroke=col, sw=2.4)
        s.text(lx + 27, ly, label, role="math", size=17, fill="ink2")
        lx += 27 + Svg.width(label, "math", 17) + 22
    return s


# what a run measures -------------------------------------------------------
def measure(theme: str) -> Svg:
    model, rows = spectra()
    layer = next(r for r in rows if r["name"] == ANATOMY_LAYER)
    W, H = 1200, 400
    s = Svg(W, H, theme, "What one run measures",
            "Four outputs, left to right. Weights: every supported matrix of a checkpoint pinned to a commit. "
            "Spectra: squared singular values with a power-law tail fit. Fingerprints: seeded sketches and hashes "
            "of sampled row blocks. Zoo: packages built into Parquet tables that are queried offline.")
    frame(s)
    pulse_ring(s)
    cw, ch, gap, x0, y0 = 266, 336, 24, 32, 32
    items = [
        ("01", "Weights", ["Every supported matrix at a", "pinned commit; skipped ones", "keep their reason."]),
        ("02", "Spectra", ["Squared singular values, a", "power-law tail fit, norms,", "rank and entropy."]),
        ("03", "Fingerprints", ["Seeded 64 × 64 sketches and", "hashes of sampled row blocks,", "for lineage and similarity."]),
        ("04", "Zoo", ["One package per model, built", "into Parquet tables you query", "offline."]),
    ]
    for i, (num, title, body) in enumerate(items):
        x = x0 + i * (cw + gap)
        card(s, x, y0, cw, ch)
        card(s, x, y0, cw, ch, fill="none", stroke="gold", cls="lit", attrs=f'style="--t:{0.5 * i:.1f}s;--d:4s"')
        vx, vy, vw, vh = x + 16, y0 + 16, cw - 32, 168
        s.rect(vx, vy, vw, vh, rx=8, fill="panel")
        if i == 0:
            heat_tile(s, vx + 33, vy + 22, 12, 8, 14, seed=7)
            s.text(vx + vw / 2, vy + vh - 14, [("W", {}), (" ∈ ", {"role": "op"}), ("ℝ", {"role": "op"}),
                                               ("m×n", {"sup": True, "role": "math"})],
                   role="math", size=19, fill="ink2", anchor="middle")
        elif i == 1:
            logs = np.log10(layer["eigs"])
            edges = np.linspace(logs.min(), logs.max(), 29)
            counts, _ = np.histogram(logs, edges)
            bw = (vw - 40) / len(counts)
            xm = math.log10(layer["xmin"])
            for k, n in enumerate(counts):
                bh = 104 * n / counts.max()
                col = "gold" if edges[k] >= xm else "blue"
                s.rect(vx + 20 + k * bw, vy + 128 - bh, bw - 2, bh, rx=1.5, fill=col, opacity=0.9 if col == "gold" else 0.55)
            s.line(vx + 16, vy + 129, vx + vw - 16, vy + 129, stroke="line2")
            s.text(vx + vw - 18, vy + 30, [("α", {}), (" = ", {"role": "mathr"}), (f"{layer['alpha']:.2f}", {"role": "mathr"})],
                   role="math", size=19, fill="gold_t", anchor="end")
            s.text(vx + vw / 2, vy + vh - 14, [("log", {"role": "mathr"}), ("10", {"sub": True, "role": "mathr"}), (" λ", {})],
                   role="math", size=17, fill="ink3", anchor="middle")
        elif i == 2:
            heat_tile(s, vx + 20, vy + 20, 8, 8, 13.5, seed=11, gap=1.5, rank=3)
            for k, (lab, val) in enumerate((("sketch", "64 × 64"), ("rows", "256"), ("hash", "SHA-256"))):
                yy = vy + 34 + k * 38
                s.text(vx + 140, yy, lab, role="sans", size=14, fill="ink3")
                s.text(vx + 140, yy + 19, val, role="mono", size=15, fill="ink")
            s.text(vx + vw / 2, vy + vh - 14, [("S", {}), (" = ", {"role": "mathr"}), ("L B R", {}),
                                               ("⊤", {"sup": True, "role": "op"})], role="math", size=19, fill="ink2", anchor="middle")
        else:
            cat = catalog()
            xs = (cat.alpha_median - 1.75) / 1.3
            ys = (np.log(cat.stable_rank_median) - np.log(25)) / (np.log(150) - np.log(25))
            pos = {m: (vx + 20 + a * (vw - 40), vy + 124 - b * 104) for m, a, b in zip(cat.model_id, xs, ys)}
            for m, parent in zip(cat.model_id, cat.parent):
                if parent and parent != m and parent in pos:
                    (x1, y1), (x2, y2) = pos[parent], pos[m]
                    s.line(x1, y1, x2, y2, stroke="ink3", sw=0.9, opacity=0.7)
            for k, (m, arch) in enumerate(zip(cat.model_id, cat.arch)):
                _, col = family(arch)
                mx_, my_ = pos[m]
                s.circle(mx_, my_, 7, fill=col, opacity=0.2, cls="breathe", attrs=f'style="--t:{-0.41 * k:.2f}s;--d:{4 + (k % 4) * 0.7:.1f}s"')
                s.circle(mx_, my_, 3.4, fill=col)
            s.text(vx + vw / 2, vy + vh - 14, "find · similar · diff", role="mono", size=15, fill="ink2", anchor="middle")
        s.text(x + 20, y0 + 222, ROMAN[num], role="serifi", size=21, fill=STAGE[num])
        s.text(x + 20, y0 + 256, title, role="serif", size=32, fill="ink")
        for k, line in enumerate(body):
            s.text(x + 20, y0 + 284 + k * 21, line, role="sans", size=16, fill="ink2")
        if i < 3:
            ax = x + cw + 3
            s.path(f"M{ax} {y0 + 100}h{gap - 6}", stroke="line2", sw=1.4)
            s.arrowhead(ax + gap - 5, y0 + 100, 0, 6, fill="line2")
            s.comet(f"M{x + cw - 40} {y0 + 100}h{gap + 80}", "gold", dur=4, delay=0.5 * i + 0.2, sw=2)
    return s


# quick start terminal ------------------------------------------------------
def terminal(theme: str) -> Svg:
    W, H = 1200, 392
    s = Svg(W, H, theme, "Quick start in four commands",
            "A terminal session. Prepare pins each model to a commit and marks rows ready or held without using "
            "a GPU. Launch measures GPU spectra and CPU fingerprints and publishes one package per model. "
            "zoo.cli build writes Parquet tables and a retrieval index. zoo_keeper.cli similar finds the nearest "
            "models by weight fingerprint, offline.")
    c = s.c
    frame(s, sky=False, n_stars=0)
    s.rect(0.5, 0.5, W - 1, 44, rx=16, fill="lift")
    s.rect(0.5, 30, W - 1, 15, fill="lift")
    s.line(0.5, 44.5, W - 0.5, 44.5, stroke="line")
    for k, col in enumerate(("ember", "gold", "green")):
        s.circle(28 + k * 20, 22.5, 5.5, fill=col, opacity=0.85)
    s.text(W / 2, 28, "~/Model-Zoo", role="texti", size=16, fill="ink3", anchor="middle")
    script = [
        ("cmd", ["python esd_experiment/run_experiment.py --model_list models.csv \\",
                 "    --output_dir analysis_runs/phase2/my_run --limit 5 --prepare_only"]),
        ("out", "pins every row to a commit · ready rows launch, held rows keep a reason · no GPU"),
        ("cmd", ["python esd_experiment/run_experiment.py \\",
                 "    --model_list analysis_runs/phase2/my_run/models.csv \\",
                 "    --output_dir analysis_runs/phase2/my_run --gpus 0 1 2 3"]),
        ("out", "GPU spectra and CPU fingerprints · one immutable package per model"),
        ("cmd", ["python -m zoo.cli build"]),
        ("out", "Parquet tables and a retrieval index in $ZOO_DATA/builds/current"),
        ("cmd", ["python -m zoo_keeper.cli similar openai-community/gpt2 -k 5"]),
        ("out", "nearest models by weight fingerprint · no weights or GPU needed"),
    ]
    size, lh, x0, y = 15.5, 23.5, 34, 82
    char = Svg.width("m", "mono", size)
    cycle, cps, t = 19.0, 34.0, 0.4
    css = []
    lines = []
    for kind, content in script:
        if kind == "cmd":
            for k, text in enumerate(content):
                full = ("$ " if k == 0 else "  ") + text
                dur = len(full) / cps
                lines.append(("cmd", full, y, t, dur, k == len(content) - 1))
                t += dur + 0.05
                y += lh
            t += 0.35
        else:
            lines.append(("out", content, y, t, 0.0, False))
            t += 0.9
            y += lh + 10
    end = t + 0.2
    hold_end = cycle - 0.9
    for n, (kind, text, ly, start, dur, last) in enumerate(lines):
        if kind == "cmd":
            s.text(x0, ly, [("$", {"fill": "gold_t"}), (text[1:], {})] if text.startswith("$") else text,
                   role="mono", size=size, fill="ink")
            width = Svg.width(text, "mono", size) + 6
            a, b = 100 * start / cycle, 100 * (start + dur) / cycle
            off = b + (1.4 if last else 0.0)
            steps = max(1, len(text))
            css.append(f"@keyframes ty{n}{{0%,{a:.2f}%{{transform:translateX(0)}}"
                       f"{b:.2f}%,100%{{transform:translateX({width:.1f}px)}}}}"
                       f".ty{n}{{transform:translateX({width:.1f}px);animation:ty{n} {cycle}s steps({steps},end) infinite}}"
                       f"@keyframes cu{n}{{0%,{max(a - 0.01, 0):.2f}%{{opacity:0}}{a:.2f}%,{off:.2f}%{{opacity:1}}"
                       f"{off + 0.01:.2f}%,100%{{opacity:0}}}}"
                       f".cu{n}{{opacity:0;animation:cu{n} {cycle}s linear infinite}}")
            s.add(f'<g class="ty{n}"><rect x="{x0 - 2}" y="{fmt(ly - size - 2)}" width="{fmt(W - x0 - 20)}" height="{fmt(lh + 1)}" '
                  f'fill="{c["panel"]}"/><rect class="cu{n}" x="{x0 - 1}" y="{fmt(ly - size + 1)}" width="{fmt(char)}" '
                  f'height="{fmt(size + 3)}" fill="{c["gold"]}"/></g>')
        else:
            a = 100 * start / cycle
            css.append(f"@keyframes ou{n}{{0%,{a:.2f}%{{opacity:0}}{a + 2:.2f}%,100%{{opacity:1}}}}"
                       f".ou{n}{{animation:ou{n} {cycle}s ease-out infinite}}")
            s.open(f'class="ou{n}"')
            sparkle(s, x0 + 8, ly - 5, 7, ("gold", "teal", "ember", "blue")[n % 4])
            s.text(x0 + 24, ly, text, role="texti", size=size + 1, fill="ink2")
            s.close()
    css.append(f"@keyframes wipe{{0%,{100 * hold_end / cycle:.2f}%{{opacity:1}}{100 * (cycle - 0.25) / cycle:.2f}%,100%{{opacity:0}}}}"
               f".wipe{{animation:wipe {cycle}s ease-in infinite}}")
    s.style("".join(css))
    s.define("termclip", f'<clipPath id="termclip"><rect x="2" y="46" width="{W - 4}" height="{H - 48}"/></clipPath>')
    s.body.insert(next(i for i, b in enumerate(s.body) if "~/Model-Zoo" in b) + 1,
                  '<g clip-path="url(#termclip)"><g class="wipe">')
    s.close()
    s.close()
    s.open('clip-path="url(#frameclip)"')
    nebula(s, [(0.92, 0.2, 0.3, "neb1", 0.16), (1.02, 1.0, 0.28, "neb2", 0.12), (0.45, 1.1, 0.3, "neb3", 0.08)])
    stars(s, 34, seed=9, box=(860, 60, 1190, 384), bright=3)
    s.close()
    assert end < hold_end, (end, hold_end)
    return s


# architecture --------------------------------------------------------------
def architecture(theme: str) -> Svg:
    W, H = 1200, 600
    s = Svg(W, H, theme, "Architecture: plan, own resources, execute, publish",
            "The runner reads the pinned models.csv, runs preflight and resume checks and routes oversized models "
            "to later row-block jobs. The dispatcher owns GPU reservations, process groups and CPU thread budgets "
            "and reloads live policy on SIGHUP. Each isolated worker loads one model, computes spectra on GPU "
            "threads while a CPU sidecar fingerprints the same weights, and publishes one package. A supervisor "
            "watches heartbeat, stage time, RAM and cache. After the worker's process group exits, its private "
            "cache is removed and its GPUs return to the pool. An unconfirmed exit keeps the cache and reservation "
            "and stops new dispatch.")
    frame(s)
    yA, yB = 168, 330
    # input
    card(s, 28, 214, 124, 172)
    hf = place_icon(s, "hugging-face", 28 + 62 - 15, 230, 28)
    s.text(90, 286, "models.csv", role="mono", size=14, fill="ink", anchor="middle")
    for k, (lab, col) in enumerate((("ready", "green_t"), ("ready", "green_t"), ("held", "gold_t"))):
        s.text(46, 316 + k * 22, "●", role="sans", size=10, fill=col)
        s.text(60, 316 + k * 22, lab, role="sans", size=14, fill="ink2")
    # runner
    rx, rw = 192, 220
    card(s, rx, 186, rw, 228)
    kicker(s, rx + 18, 214, "01", "Plan")
    s.text(rx + 18, 250, "Runner", role="serif", size=30, fill="ink")
    s.text(rx + 18, 275, "run_experiment.py", role="mono", size=14, fill="ink3")
    for k, line in enumerate(["pins and preflight", "resumes compatible work", "routes oversized models", "to row-block jobs"]):
        s.text(rx + 18, 310 + k * 22, line, role="sans", size=15.5, fill="ink2" if k < 3 else "ink3")
    # dispatcher (the core)
    dx, dw = 452, 236
    s.rect(dx, 186, dw, 228, rx=10, fill="tile", stroke="tile")
    s.text(dx + 18, 214, [("II", {"role": "serifi", "size": 20, "fill": "ember_t"}), ("   Own resources", {})],
           role="texti", size=16, fill="tile_ink", opacity=0.85)
    s.text(dx + 18, 250, "Dispatcher", role="serif", size=30, fill="tile_ink")
    s.text(dx + 18, 275, "gputracker/", role="mono", size=14, fill="tile_ink", opacity=0.7)
    for k, line in enumerate(["idle checks", "GPU reservations", "process groups", "CPU thread budget"]):
        s.text(dx + 18, 310 + k * 22, line, role="sans", size=15.5, fill="tile_ink", opacity=0.92)
    # workers
    wx, ww = 728, 262
    for k, (yy, name) in enumerate(((150, "Worker A"), (322, "Worker B"))):
        card(s, wx, yy, ww, 148)
        kicker(s, wx + 18, yy + 26, "03", "Execute in isolation")
        s.text(wx + 18, yy + 60, name, role="serif", size=26, fill="ink")
        place_icon(s, "pytorch", wx + ww - 36, yy + 14, 22)
        s.text(wx + ww - 18, yy + 60, "worker.py", role="mono", size=13, fill="ink3", anchor="end")
        s.rect(wx + 16, yy + 76, 112, 56, rx=7, fill="lift", stroke="line")
        s.text(wx + 26, yy + 98, "GPU threads", role="sans", size=13.5, fill="blue_t")
        s.text(wx + 26, yy + 119, "net_esd/", role="mono", size=13, fill="ink2")
        s.rect(wx + 136, yy + 76, 110, 56, rx=7, fill="lift", stroke="line")
        s.text(wx + 146, yy + 98, "CPU sidecar", role="sans", size=13.5, fill="teal_t")
        s.text(wx + 146, yy + 119, "evidence/", role="mono", size=13, fill="ink2")
    # package
    px, pw = 1030, 142
    card(s, px, 230, pw, 140)
    kicker(s, px + 16, 258, "04", "Publish")
    s.text(px + 16, 292, "Package", role="serif", size=26, fill="ink")
    s.text(px + 16, 316, "one per model,", role="sans", size=14, fill="ink2")
    s.text(px + 16, 336, "immutable", role="sans", size=14, fill="ink2")
    s.text(px + 16, 356, "$ZOO_DATA/", role="mono", size=12.5, fill="ink3")
    # supervision rail and live policy
    s.rect(dx, 64, wx + ww - dx, 48, rx=24, fill="lift", stroke="line")
    s.text(dx + 22, 94, "Supervisor", role="sansb", size=15, fill="gold_t")
    s.text(dx + 116, 94, "heartbeat · stage time · RAM · cache", role="sans", size=15, fill="ink2")
    e0 = dx + 116 + Svg.width("heartbeat · stage time · RAM · cache", "sans", 15) + 26
    beats = max(1, int((wx + ww - 24 - e0 - 20) // 54))
    ecg = f"M{e0:.1f} 88h16" + "l6-12l7 22l6-10h35" * beats
    s.path(ecg, stroke="gold", sw=1.4, opacity=0.6)
    s.comet(ecg, "gold", dur=3.2, delay=0, sw=2)
    for xx in (dx + dw / 2, wx + 60, wx + ww - 60):
        s.line(xx, 112, xx, 150 if xx > wx else 186, stroke="gold", sw=1.2, dash="3 4", opacity=0.8)
    cw_ = chip(s, 192, 64, "gpu_config.json + SIGHUP", size=13.5)
    s.text(192, 124, "live policy, reloaded", role="sans", size=13.5, fill="ink3")
    s.text(192, 143, "at a safe point", role="sans", size=13.5, fill="ink3")
    pol = f"M{192 + cw_ + 6:.1f} 78H{dx - 30}Q{dx - 12} 78 {dx - 12} 96V150Q{dx - 12} 168 {dx + 6} 168H{dx + 30}V186"
    s.path(pol, stroke="gold", sw=1.2, opacity=0.8)
    s.arrowhead(dx + 30, 186, 90, 6, fill="gold")
    # flow
    flows = [f"M152 300H{rx}", f"M{rx + rw} 300H{dx}", f"M{dx + dw} 300C{dx + dw + 22} 300 {wx - 24} 224 {wx} 224",
             f"M{dx + dw} 300C{dx + dw + 22} 300 {wx - 24} 396 {wx} 396",
             f"M{wx + ww} 224C{wx + ww + 22} 224 {px - 22} 300 {px} 300", f"M{wx + ww} 396C{wx + ww + 22} 396 {px - 22} 300 {px} 300"]
    for k, d in enumerate(flows):
        s.path(d, stroke="line2", sw=1.4)
        s.comet(d, "gold", dur=3.4, delay=0.35 * (k if k < 3 else k - 1), sw=2.2)
    for x2, y2 in ((rx, 300), (dx, 300), (wx, 224), (wx, 396), (px, 300)):
        s.arrowhead(x2, y2, 0, 6.5, fill="line2")
    # release loop
    loop = f"M{wx + ww / 2} 470V500Q{wx + ww / 2} 520 {wx + ww / 2 - 20} 520H{dx + dw / 2 + 20}Q{dx + dw / 2} 520 {dx + dw / 2} 500V414"
    s.path(loop, stroke="green", sw=1.6, opacity=0.8)
    s.comet(loop, "green", dur=3.6, delay=1.2, sw=2.4)
    s.arrowhead(dx + dw / 2, 416, -90, 7, fill="green")
    steps = ["confirm process-group exit", "remove private cache", "release GPUs"]
    lx = dx + dw / 2 + 22
    for k, st in enumerate(steps):
        s.circle(lx + 9, 549, 9, fill="green", opacity=0.18)
        s.text(lx + 9, 554, str(k + 1), role="sansb", size=12.5, fill="green_t", anchor="middle")
        s.text(lx + 25, 554, st, role="sans", size=14.5, fill="green_t")
        lx += 25 + Svg.width(st, "sans", 14.5) + 26
    s.text(rx, 520, [("Capacity returns", {"role": "sansb", "fill": "ink"}), (" only after", {})], role="sans", size=15, fill="ink2")
    s.text(rx, 541, "confirmed cleanup.", role="sans", size=15, fill="ink2")
    s.text(px, 402, "Exit unconfirmed?", role="sansb", size=14, fill="red_t")
    s.text(px, 422, "Keep cache and", role="sans", size=14, fill="ink2")
    s.text(px, 440, "reservation; stop", role="sans", size=14, fill="ink2")
    s.text(px, 458, "new dispatch.", role="sans", size=14, fill="ink2")
    return s


# GPU parallelism -----------------------------------------------------------
def gpus(theme: str) -> Svg:
    W, H = 1200, 476
    s = Svg(W, H, theme, "Two levels of parallelism",
            "Across models, isolated worker processes each reserve their own GPUs: worker A holds GPUs 0 and 1, "
            "worker B holds GPUs 2 and 3, and model C waits for capacity. Within a model, the layer queue is "
            "sorted by estimated cost and dispatched to one fixed-device thread per GPU; metrics return in the "
            "original layer order.")
    frame(s)
    # queue
    card(s, 28, 40, 236, 404)
    kicker(s, 46, 70, "01", "Model queue")
    for k, (name, col, note) in enumerate((("Model A", "blue", "running"), ("Model B", "green", "running"),
                                           ("Model C", "gold", "waits for idle GPUs"))):
        yy = 92 + k * 92
        s.rect(46, yy, 200, 70, rx=9, fill="lift", stroke=col if k < 2 else "line", sw=1.2,
               attrs='stroke-dasharray="5 4"' if k == 2 else "")
        s.circle(66, yy + 27, 5, fill=col)
        s.text(80, yy + 32, name, role="sansb", size=17, fill="ink")
        s.text(66, yy + 55, note, role="sans", size=13.5, fill="ink3")
        if k == 2:
            s.circle(66, yy + 27, 9, fill=col, opacity=0.25, cls="breathe", attrs='style="--d:2.4s"')
    s.text(46, 392, "Admission: two idle checks", role="sans", size=13.5, fill="ink3")
    s.text(46, 412, "5 s apart, under 500 MB used", role="sans", size=13.5, fill="ink3")
    # workers
    lanes = [("Worker A", "blue", "blue_t", (0, 1), 40), ("Worker B", "green", "green_t", (2, 3), 250)]
    for name, col, colt, ids, yy in lanes:
        card(s, 300, yy, 872, 194)
        s.rect(300, yy, 872, 40, rx=10, fill=col, opacity=0.14)
        s.text(320, yy + 26, name, role="sansb", size=16, fill=colt)
        s.text(410, yy + 26, "own process group · private cache · CUDA_VISIBLE_DEVICES", role="sans", size=14, fill="ink2")
        place_icon(s, "nvidia", 1172 - 18 - 76, yy + 13, 14)
        kicker(s, 320, yy + 72, "02", "Layer queue, largest cost first")
        widths = [96, 74, 58, 44, 32, 22, 14]
        bx = 320
        for k, bw in enumerate(widths):
            s.rect(bx, yy + 92, bw, 30, rx=4, fill=col, opacity=f"{0.95 - k * 0.11:.2f}")
            bx += bw + 6
        s.text(320, yy + 150, "matrices sorted by SVD cost, then", role="sans", size=13.5, fill="ink3")
        s.text(320, yy + 168, "split across this worker's GPUs", role="sans", size=13.5, fill="ink3")
        gx = 790
        for k, gid in enumerate(ids):
            gy = yy + 62 + k * 62
            s.rect(gx, gy, 196, 48, rx=8, fill="lift", stroke=col, sw=1.1)
            s.rect(gx + 12, gy + 12, 24, 24, rx=4, fill="none", stroke=col, sw=1.4)
            for p in range(3):
                s.line(gx + 17 + p * 7, gy + 8, gx + 17 + p * 7, gy + 12, stroke=col, sw=1.2)
                s.line(gx + 17 + p * 7, gy + 36, gx + 17 + p * 7, gy + 40, stroke=col, sw=1.2)
            s.text(gx + 48, gy + 30, [(f"GPU {gid}", {"role": "sansb"}), ("  →  ", {}), (f"cuda:{k}", {"role": "mono"})],
                   role="sans", size=15, fill="ink")
            d_in = f"M{bx} {yy + 107}C{bx + 60} {yy + 107} {gx - 60} {gy + 24} {gx} {gy + 24}"
            s.path(d_in, stroke="line2", sw=1.3)
            s.arrowhead(gx, gy + 24, 0, 6, fill="line2")
            s.comet(d_in, col, dur=2.8, delay=0.6 * k + (0 if col == "blue" else 1.1), sw=2.2)
            d_out = f"M{gx + 196} {gy + 24}C{gx + 216} {gy + 24} {gx + 216} {yy + 107} {gx + 236} {yy + 107}"
            s.path(d_out, stroke="line2", sw=1.3)
        s.text(1030, yy + 102, "layer metrics", role="sansb", size=14.5, fill="ink")
        s.text(1030, yy + 122, "original order", role="sans", size=13.5, fill="ink3")
    return s


# worker lifecycle ----------------------------------------------------------
def lifecycle(theme: str) -> Svg:
    W, H = 1200, 360
    s = Svg(W, H, theme, "Worker lifecycle and runtime signals",
            "A worker loads, analyzes and finalizes one model under heartbeat, stage-time, RAM and cache "
            "monitoring. Cleanup confirms process-group exit, removes the private cache and then releases GPUs. "
            "SIGHUP reloads live policy at a safe point, SIGUSR1 finishes active work without starting new "
            "models, and SIGINT or SIGTERM terminates, confirms exit and cleans up. An unconfirmed exit keeps "
            "evidence and stops dispatch.")
    frame(s)
    # monitor
    s.rect(28, 30, 640, 46, rx=23, fill="lift", stroke="line")
    s.text(50, 59, "Monitor", role="sansb", size=15, fill="gold_t")
    s.text(124, 59, "heartbeat · stage time · RAM · cache", role="sans", size=15, fill="ink2")
    ecg = "M400 53h34l5-11l6 20l5-9h46l5-11l6 20l5-9h46l5-11l6 20l5-9h56"
    s.path(ecg, stroke="gold", sw=1.4, opacity=0.65)
    s.comet(ecg, "gold", dur=2.6, sw=2)
    stages = [("Load", "weights at the pin"), ("Analyze", "spectra and fingerprints"), ("Finalize", "publish the package")]
    x = 28
    for k, (name, note) in enumerate(stages):
        w = 196
        s.rect(x, 112, w, 86, rx=10, fill="tile" if k == 1 else "panel2", stroke="tile" if k == 1 else "line")
        ink = "tile_ink" if k == 1 else "ink"
        s.text(x + 18, 148, name, role="serif", size=26, fill=ink)
        s.text(x + 18, 176, note, role="sans", size=13.5, fill=ink, opacity=0.75)
        s.line(x + w / 2, 76, x + w / 2, 112, stroke="gold", sw=1.1, dash="3 4", opacity=0.75)
        if k < 2:
            arrow(s, x + w + 4, 155, x + w + 22, 155, stroke="line2", head=6)
        x += w + 26
    # cleanup
    cx = 700
    card(s, cx, 30, 472, 168, fill="panel2", stroke="green")
    s.text(cx + 20, 60, "Confirmed cleanup, in order", role="sansb", size=15, fill="green_t")
    for k, st in enumerate(["Confirm process-group exit", "Remove the private worker cache", "Release the GPU reservation"]):
        s.circle(cx + 32, 92 + k * 32, 11, fill="green", opacity=0.16)
        s.text(cx + 32, 97 + k * 32, str(k + 1), role="sansb", size=13.5, fill="green_t", anchor="middle")
        s.text(cx + 54, 97 + k * 32, st, role="sans", size=15.5, fill="ink")
    s.line(cx + 330, 78, cx + 330, 176, stroke="line")
    s.text(cx + 348, 96, "Unconfirmed?", role="sansb", size=14, fill="red_t")
    s.text(cx + 348, 118, "Keep evidence", role="sans", size=14, fill="ink2")
    s.text(cx + 348, 138, "and reservation,", role="sans", size=14, fill="ink2")
    s.text(cx + 348, 158, "stop dispatch.", role="sans", size=14, fill="ink2")
    arrow(s, 668, 155, cx - 6, 155, stroke="green", sw=1.4, head=6)
    s.comet(f"M600 155H{cx}", "green", dur=3, delay=1, sw=2)
    # signals
    s.line(28, 236, W - 28, 236, stroke="line")
    sigs = [("SIGHUP", "Reload policy", "edit gpu_config.json; applied at a safe point", "gold_t"),
            ("SIGUSR1", "Finish active work", "running workers complete; nothing new starts", "blue_t"),
            ("SIGINT · SIGTERM", "Stop and clean up", "terminate, confirm exit, then clean up", "red_t")]
    for k, (sig, what, note, col) in enumerate(sigs):
        x = 28 + k * 392
        s.text(x, 272, sig, role="mono", size=16, fill=col)
        s.text(x, 300, what, role="sansb", size=16, fill="ink")
        s.text(x, 324, note, role="sans", size=14, fill="ink3")
    return s


# anatomy of a spectrum -----------------------------------------------------
def spectrum(theme: str) -> Svg:
    model, rows = spectra()
    r = next(r for r in rows if r["name"] == ANATOMY_LAYER)
    W, H = 1200, 660
    s = Svg(W, H, theme, f"Anatomy of one measured spectrum: {model}, {r['name']}",
            f"Histogram of the {len(r['eigs'])} squared singular values of a {r['M']} by {r['N']} matrix on a log "
            f"scale. A Marchenko-Pastur curve shows an i.i.d. matrix of the same shape and norm for reference. "
            f"The tail above the median eigenvalue is fitted by a power law with alpha {r['alpha']:.2f} and "
            f"Kolmogorov-Smirnov distance {r['D']:.3f}. A log-log plot of the tail shows the fitted line. "
            f"A strip below shows alpha for all {len(rows)} matrices of the model in depth order.")
    frame(s)
    logs = np.log10(r["eigs"])
    # panel A: histogram with MP reference
    ax0, ax1, ay0, ay1 = 70, 680, 122, 380
    s.text(40, 52, [("A", {"role": "sansb", "fill": "ink"}), ("   Eigenvalue density", {})], role="sans", size=16, fill="ink2")
    s.text(40, 74, f"{model} · {r['name']} · {r['M']} × {r['N']}", role="mono", size=13, fill="ink3")
    lo, hi = -1.4, 1.9
    sx = lambda v: ax0 + (v - lo) / (hi - lo) * (ax1 - ax0)
    edges = np.linspace(lo, hi, 67)
    counts, _ = np.histogram(logs, edges)
    dens = counts / (counts.sum() * (edges[1] - edges[0]))
    u = np.linspace(lo, hi, 600)
    mp, (lam_m, lam_p) = marchenko_pastur_log10(u, r["M"], r["N"], r["norm"])
    ymax = max(dens.max(), mp.max()) * 1.08
    sy = lambda v: ay1 - v / ymax * (ay1 - ay0)
    xm = math.log10(r["xmin"])
    s.rect(sx(xm), ay0 - 6, ax1 - sx(xm), ay1 - ay0 + 6, fill="gold", opacity=0.07 if theme == "dark" else 0.055)
    for k, d in enumerate(dens):
        if d <= 0:
            continue
        x1, x2 = sx(edges[k]), sx(edges[k + 1])
        tail = edges[k] >= xm - 1e-9
        s.rect(x1 + 0.6, sy(d), x2 - x1 - 1.2, ay1 - sy(d), rx=1, fill="gold" if tail else "blue",
               opacity=0.9 if tail else 0.5)
    mp_pts = [(sx(a), sy(b)) for a, b in zip(u, mp)]
    s.path(poly(simplify(mp_pts, 0.08)), stroke="teal", sw=2.2)
    s.line(ax0, ay1, ax1, ay1, stroke="line2")
    for v in (-1, 0, 1):
        s.line(sx(v), ay1, sx(v), ay1 + 6, stroke="line2")
        s.text(sx(v), ay1 + 24, str(v).replace("-", "−"), role="mathr", size=16, fill="ink3", anchor="middle")
    s.text(ax1, ay1 + 26, [("log", {"role": "mathr"}), ("10", {"sub": True, "role": "mathr"}), (" λ", {})], role="math",
           size=18, fill="ink2", anchor="end")
    s.line(sx(xm), ay0 - 30, sx(xm), ay1, stroke="gold", sw=1.4, dash="5 4")
    s.text(sx(xm) + 8, ay0 - 14, [("λ", {}), ("min", {"sub": True, "role": "mathr"}),
                                  ("   median eigenvalue, the tail starts here", {"role": "sans", "size": 13.5, "fill": "ink3"})],
           role="math", size=19, fill="gold_t")
    lmax = math.log10(r["spectral_norm"])
    s.line(sx(lmax), ay1 - 40, sx(lmax), ay1, stroke="ember", sw=1.6)
    s.circle(sx(lmax), ay1 - 40, 3.5, fill="ember")
    s.text(sx(lmax) - 6, ay1 - 52, [("λ", {}), ("max", {"sub": True, "role": "mathr"})], role="math", size=19,
           fill="ember_t", anchor="middle")
    peak = int(np.argmax(mp))
    lx, ly = sx(u[peak]) - 92, sy(mp[peak]) + 18
    s.text(lx - 54, ly + 6, "Marchenko–Pastur", role="sansb", size=14, fill="teal_t", anchor="end")
    s.text(lx - 54, ly + 25, "i.i.d. matrix, same shape", role="sans", size=13.5, fill="ink3", anchor="end")
    s.text(lx - 54, ly + 43, "and norm, for reference", role="sans", size=13.5, fill="ink3", anchor="end")
    s.line(lx - 46, ly + 2, sx(u[peak]) - 30, sy(mp[peak] * 0.8), stroke="teal", sw=1, opacity=0.7)
    s.circle(sx(lmax), ay1 - 40, 9, fill="ember", opacity=0.25, cls="breathe", attrs='style="--d:3s"')

    # panel B: tail on log-log axes with the fitted line
    bx0, bx1, by0, by1 = 790, 1160, 96, 380
    s.text(760, 52, [("B", {"role": "sansb", "fill": "ink"}), ("   Tail fit", {})], role="sans", size=16, fill="ink2")
    s.text(760, 74, f"{r['n_tail']} eigenvalues at or above the cutoff", role="mono", size=13, fill="ink3")
    tail = r["eigs"][r["eigs"] >= r["xmin"]]
    n = len(tail)
    ccdf = 1 - np.arange(n) / n
    tl, th = math.log10(tail.min()), math.log10(tail.max()) + 0.05
    cl, ch = math.log10(ccdf.min()) - 0.1, 0.05
    tx = lambda v: bx0 + (v - tl) / (th - tl) * (bx1 - bx0)
    ty = lambda v: by1 - (v - cl) / (ch - cl) * (by1 - by0)
    for k in range(0, n, 2):
        s.circle(tx(math.log10(tail[k])), ty(math.log10(ccdf[k])), 2.2, fill="gold", opacity=0.85)
    fit_x = np.linspace(tl, th - 0.05, 40)
    fit_y = -(r["alpha"] - 1) * (fit_x - math.log10(r["xmin"]))
    pts = [(tx(a), ty(b)) for a, b in zip(fit_x, fit_y) if b >= cl]
    s.path(poly(pts), stroke="ink", sw=1.8, dash="7 5")
    s.line(bx0, by1, bx1, by1, stroke="line2")
    s.line(bx0, by0, bx0, by1, stroke="line2")
    s.text(bx1, by1 + 26, [("log", {"role": "mathr"}), ("10", {"sub": True, "role": "mathr"}), (" λ", {})], role="math",
           size=18, fill="ink2", anchor="end")
    s.text(bx0 - 12, by0 + 8, [("P", {}), ("(Λ ≥ λ)", {"role": "mathr"})], role="math", size=17, fill="ink2", anchor="end",
           attrs=f'transform="rotate(-90 {bx0 - 12} {by0 + 8})"')
    s.text(bx0 + 190, by0 + 28, [("α̂", {}), (" = ", {"role": "mathr"}), (f"{r['alpha']:.2f}", {"role": "mathr"})],
           role="math", size=24, fill="gold_t")
    s.text(bx0 + 190, by0 + 54, [("D", {}), (" = ", {"role": "mathr"}), (f"{r['D']:.3f}", {"role": "mathr"}),
                                 ("  KS distance", {"role": "sans", "size": 13.5, "fill": "ink3"})], role="math", size=19, fill="ink2")
    s.text(bx0 + 22, by1 - 22, "slope −(α − 1)", role="sans", size=13.5, fill="ink3")

    # strip: alpha by depth
    sy0, sy1 = 470, 600
    s.line(40, 420, W - 40, 420, stroke="line")
    s.text(40, 452, [("C", {"role": "sansb", "fill": "ink"}), ("   Fitted α for every matrix, input to output", {})],
           role="sans", size=16, fill="ink2")
    kinds = [("query_key_value", "attention QKV", "blue"), ("attention.dense", "attention out", "ember"),
             ("dense_h_to_4h", "MLP up", "gold"), ("dense_4h_to_h", "MLP down", "teal"), ("embed", "embedding", "ink2")]
    amin, amax = 1.4, 5.1
    yv = lambda a: sy1 - (a - amin) / (amax - amin) * (sy1 - sy0)
    xs0, xs1 = 90, W - 60
    for a in (2, 3, 4, 5):
        s.line(xs0, yv(a), xs1, yv(a), stroke="grid")
        s.text(xs0 - 14, yv(a) + 5, str(a), role="mathr", size=15, fill="ink3", anchor="end")
    s.text(xs0 - 40, yv(3.5) + 6, "α", role="math", size=19, fill="ink2", anchor="middle")
    step = (xs1 - xs0) / (len(rows) - 1)
    for i, row in enumerate(rows):
        col = next(c for key, _, c in kinds if key in row["name"])
        x = xs0 + i * step
        s.line(x, sy1, x, yv(row["alpha"]), stroke=col, sw=1, opacity=0.35)
        s.circle(x, yv(row["alpha"]), 4.6 if row["name"] != ANATOMY_LAYER else 6.5, fill=col,
                 stroke="ink" if row["name"] == ANATOMY_LAYER else None, sw=1.6)
    lx = 470
    for _, label, col in kinds:
        s.circle(lx, 447, 4.6, fill=col)
        s.text(lx + 11, 452, label, role="sans", size=14, fill="ink2")
        lx += 22 + Svg.width(label, "sans", 14) + 18
    s.text(xs0, sy1 + 34, "embed_in", role="mono", size=13, fill="ink3")
    s.text(xs1, sy1 + 34, "embed_out", role="mono", size=13, fill="ink3", anchor="end")
    s.text(W / 2, sy1 + 34, "12 transformer layers", role="sans", size=13.5, fill="ink3", anchor="middle")
    return s


# fingerprints --------------------------------------------------------------
def fingerprint(theme: str) -> Svg:
    W, H = 1200, 470
    s = Svg(W, H, theme, "Weight fingerprints and lineage",
            "A deterministic 256-row block from the middle of each matrix is sketched from both sides with fixed, "
            "seeded orthonormal operators into a 64 by 64 sketch, and hashed with SHA-256 together with its dtype. "
            "Two models are compared view by view with sketch cosine and shared hashes. Lineage evidence is "
            "declared, detected or verified, and stays uncalibrated without an applicable calibrated null.")
    frame(s)
    pulse_ring(s)
    # matrix with row block
    mx, my, mw, mh = 44, 70, 150, 330
    s.text(mx, 48, [("W", {"role": "math"}), ("  one stored matrix", {})], role="sans", size=15, fill="ink2")
    s.rect(mx, my, mw, mh, rx=6, fill="lift", stroke="line")
    g = rng(3)
    for k in range(22):
        yy = my + 8 + k * 14.6
        s.rect(mx + 8, yy, mw - 16, 8, rx=2, fill="ink3", opacity=f"{0.08 + 0.14 * g.random():.2f}")
    by, bh = my + mh / 2 - 34, 68
    s.rect(mx - 4, by, mw + 8, bh, rx=6, fill="gold", opacity=0.22)
    s.rect(mx - 4, by, mw + 8, bh, rx=6, fill="none", stroke="gold", sw=1.6)
    s.text(mx + mw + 16, by + 26, "256 rows", role="mono", size=14, fill="gold_t")
    s.text(mx + mw + 16, by + 46, "from the middle", role="sans", size=13.5, fill="ink3")
    # operators
    ox = 330
    s.text(ox, 48, "Two-sided seeded sketch", role="sans", size=15, fill="ink2")
    s.text(ox, 140, [("S", {}), (" = ", {"role": "mathr"}), ("L", {"fill": "teal_t"}), (" B ", {"fill": "gold_t"}),
                     ("R", {"fill": "teal_t"}), ("⊤", {"sup": True, "role": "op", "fill": "teal_t"})],
           role="math", size=40, fill="ink")
    for k, (sym, text) in enumerate((("L, R", "orthonormal, seeded by axis size"), ("B", "the 256-row block"),
                                     ("k", "64 by default; 128 for lab rows"))):
        yy = 186 + k * 30
        s.text(ox, yy, sym, role="math", size=18, fill="ink")
        s.text(ox + 52, yy, text, role="sans", size=14.5, fill="ink3")
    s.rect(ox, 290, 270, 96, rx=8, fill="lift", stroke="line")
    s.text(ox + 16, 318, "Block hash", role="sansb", size=14, fill="ink")
    s.text(ox + 16, 342, "SHA-256 of dtype and raw bytes", role="sans", size=13.5, fill="ink3")
    s.text(ox + 16, 366, "identical blocks share a hash", role="sans", size=13.5, fill="ink3")
    flow1 = f"M{mx + mw + 4} {by + bh / 2}C{mx + mw + 60} {by + bh / 2} {ox - 50} 128 {ox - 10} 128"
    s.path(flow1, stroke="line2", sw=1.3)
    s.comet(flow1, "gold", dur=3.2, sw=2.2)
    # sketches
    tx = 660
    s.text(tx, 48, "Sketches compared view by view", role="sans", size=15, fill="ink2")
    for k, (label, seed) in enumerate((("child", 11), ("parent", 11))):
        x = tx + k * 196
        heat_tile(s, x, 72, 8, 8, 15, seed=seed if k == 0 else 12, gap=1.5, rank=3)
        if k == 1:
            heat_tile(s, x, 72, 8, 8, 15, seed=11, gap=1.5, rank=3, cls="fpfade")
        s.text(x, 210, [("S", {}), (label, {"sub": True, "role": "mathr"})], role="math", size=20, fill="ink2")
    s.style(".fpfade{opacity:.0;animation:fpf 6s ease-in-out infinite}@keyframes fpf{0%,100%{opacity:0}40%,60%{opacity:.55}}")
    s.text(tx + 158, 122, "cos", role="mathr", size=22, fill="ink", anchor="middle")
    s.line(tx + 130, 134, tx + 186, 134, stroke="line2", sw=1.3)
    s.arrowhead(tx + 130, 134, 180, 5.5, fill="line2")
    s.arrowhead(tx + 186, 134, 0, 5.5, fill="line2")
    s.text(tx + 342, 112, "sketch cosine", role="sansb", size=14, fill="ink")
    s.text(tx + 342, 132, "per view, plus", role="sans", size=13.5, fill="ink3")
    s.text(tx + 342, 151, "shared block hashes", role="sans", size=13.5, fill="ink3")
    flow2 = f"M{ox + 230} 128C{ox + 280} 128 {tx - 50} 120 {tx - 8} 120"
    s.path(flow2, stroke="line2", sw=1.3)
    s.comet(flow2, "gold", dur=3.2, delay=1.2, sw=2.2)
    # lineage tiers
    ly = 262
    s.text(tx, ly, "Lineage evidence", role="sansb", size=15, fill="ink")
    tiers = [("declared", "supplied relationship metadata", "ink2"),
             ("detected", "method-derived candidate, with statistics", "blue_t"),
             ("verified", "checked under the method's rules", "green_t")]
    for k, (tier, note, col) in enumerate(tiers):
        yy = ly + 18 + k * 40
        w = chip(s, tx, yy, tier, role="mono", size=14, ink=col)
        s.text(tx + 116, yy + 19, note, role="sans", size=14, fill="ink2")
    s.text(tx, ly + 152, [("uncalibrated", {"role": "mono", "fill": "gold_t"}),
                          ("  until an applicable calibrated null exists", {})], role="sans", size=14, fill="ink3")
    return s


# zoo ------------------------------------------------------------------------
def zoo(theme: str) -> Svg:
    cat = catalog()
    W, H = 1200, 580
    s = Svg(W, H, theme, "The zoo: measured models and the tables that hold them",
            f"Left: the {len(cat)} models measured in this repository's phase-2 runs, placed by median layer alpha "
            "and median stable rank and colored by architecture family; fine-tunes sit beside their families. "
            "Right: the snapshot under builds/current has seven Parquet tables and a retrieval index, queried "
            "offline with find, info, similar, family, diff and check.")
    frame(s)
    px0, px1, py0, py1 = 86, 680, 90, 470
    s.text(40, 52, f"{len(cat)} models measured in this repository", role="sans", size=16, fill="ink2")
    ax_lo, ax_hi = 1.75, 3.05
    sr_lo, sr_hi = 25, 160
    sx = lambda a: px0 + (a - ax_lo) / (ax_hi - ax_lo) * (px1 - px0)
    sy = lambda v: py1 - (math.log(v) - math.log(sr_lo)) / (math.log(sr_hi) - math.log(sr_lo)) * (py1 - py0)
    for a in (2.0, 2.5, 3.0):
        s.line(sx(a), py0, sx(a), py1, stroke="grid")
        s.text(sx(a), py1 + 24, f"{a:.1f}", role="mathr", size=15, fill="ink3", anchor="middle")
    for v in (30, 60, 120):
        s.line(px0, sy(v), px1, sy(v), stroke="grid")
        s.text(px0 - 12, sy(v) + 5, str(v), role="mathr", size=15, fill="ink3", anchor="end")
    s.text(px1, py1 + 48, [("median layer ", {"role": "sans", "size": 14}), ("α", {})], role="math", size=18, fill="ink2", anchor="end")
    s.text(40, 72, "median stable rank, log scale", role="sans", size=13.5, fill="ink3")
    pos = {m: (sx(a), sy(v)) for m, a, v in zip(cat.model_id, cat.alpha_median, cat.stable_rank_median)}
    for m, parent in zip(cat.model_id, cat.parent):
        if parent and parent != m and parent in pos:
            (x1, y1), (x2, y2) = pos[parent], pos[m]
            s.line(x1, y1, x2, y2, stroke="ink3", sw=1, dash="3 3")
    s.style(".star{animation:breathe var(--d,5s) ease-in-out infinite;animation-delay:var(--t,0s)}")
    glow = 0.26 if theme == "dark" else 0.16
    for k, (m, arch) in enumerate(zip(cat.model_id, cat.arch)):
        x, y = pos[m]
        name, col = family(arch)
        s.circle(x, y, 34, fill=s.radial(f"neb-{col}", col, glow))
    for k, (m, arch) in enumerate(zip(cat.model_id, cat.arch)):
        x, y = pos[m]
        _, col = family(arch)
        s.circle(x, y, 10, fill=col, opacity=0.16, cls="star", attrs=f'style="--t:{-0.37 * k:.2f}s;--d:{4 + (k % 5) * 0.6:.1f}s"')
        s.circle(x, y, 4.8, fill=col, stroke="panel", sw=1.2)
    pythia = cat.model_id[cat.model_id.str.contains("pythia-160m")].iloc[0]
    notes = [  # label lines, anchor point (alpha, stable rank), text origin, anchor
        (["Qwen2 and Qwen2.5 0.5B", "fine-tunes"], (2.87, 62.5), (sx(2.87), sy(62.5) + 40), "middle"),
        (["two bert-base-uncased", "fine-tunes"], (1.912, 46.26), (sx(1.912) + 18, sy(46.26) - 2), "start"),
        (["smol_llama-220M and", "two fine-tunes"], (2.61, 80), (sx(2.61) + 20, sy(80) - 4), "start"),
        ([pythia], (2.51, 39.5), (sx(2.51) + 16, sy(39.5) + 28), "start"),
    ]
    for lines, (a, v), (tx_, ty_), anchor in notes:
        x, y = sx(a), sy(v)
        if anchor == "middle":
            s.line(x, y + 12, x, ty_ - 15, stroke="ink3", sw=1)
        elif ty_ > y + 14:
            s.line(x + 5, y + 6, tx_ - 2, ty_ - 13, stroke="ink3", sw=1)
        for k, line in enumerate(lines):
            s.text(tx_, ty_ + k * 18, line, role="mono" if "/" in line else "sans", size=13.5, fill="ink2", anchor=anchor)
    lx, lyy = 40, 548
    for label, _, col in FAMILIES + [("Other", "", "ink3")]:
        s.circle(lx + 5, lyy - 5, 4.8, fill=col)
        s.text(lx + 16, lyy, label, role="sans", size=14, fill="ink2")
        lx += 16 + Svg.width(label, "sans", 14) + 22
    # snapshot
    sx0 = 730
    card(s, sx0, 30, 442, 520)
    s.text(sx0 + 22, 64, "builds/current", role="mono", size=16, fill="ink")
    s.text(sx0 + 22, 88, "seven Parquet tables, a retrieval index", role="sans", size=14, fill="ink3")
    s.text(sx0 + 22, 106, "and links to full HDF5 spectra", role="sans", size=14, fill="ink3")
    tables = [("models", "pinned identity, coverage, links"), ("views", "measured matrices, block hashes"),
              ("layers", "per-layer α, fit quality, HDF5 index"), ("coverage", "measured and skipped tensors"),
              ("lineage", "candidates, tiers and statistics"), ("scores", "measured scores, revision checks"),
              ("predictions", "kept separate; empty in v0")]
    for k, (name, note) in enumerate(tables):
        yy = 126 + k * 32
        s.rect(sx0 + 22, yy, 398, 26, rx=5, fill="lift", stroke="line")
        s.text(sx0 + 34, yy + 18, name, role="mono", size=14, fill="ink")
        s.text(sx0 + 148, yy + 18, note, role="sans", size=13.5, fill="ink3")
    qy = 370
    s.text(sx0 + 22, qy, "Query offline with Zoo-Keeper", role="sansb", size=15, fill="ink")
    for k, q in enumerate(["find", "info", "similar", "family", "diff", "check"]):
        chip(s, sx0 + 22 + (k % 3) * 136, qy + 14 + (k // 3) * 40, q, size=14, ink="gold_t" if q in ("similar", "diff") else "ink2")
    s.text(sx0 + 22, qy + 118, "Example, 2026-11-02 snapshot:", role="sans", size=13.5, fill="ink3")
    s.text(sx0 + 22, qy + 140, [("similar", {"role": "mono", "fill": "gold_t"}), ("  closest neighbor at cosine 0.999992", {})],
           role="sans", size=14, fill="ink2")
    s.text(sx0 + 22, qy + 162, [("diff", {"role": "mono", "fill": "gold_t"}), ("  151 identical views, 60 changed", {})],
           role="sans", size=14, fill="ink2")
    return s


# footer ---------------------------------------------------------------------
def footer(theme: str) -> Svg:
    model, rows = spectra()
    r = next(r for r in rows if r["name"] == ANATOMY_LAYER)
    W, H = 1200, 110
    s = Svg(W, H, theme, "", "")
    grid = np.linspace(-1.6, 2.0, 220)
    dens = kde(np.log10(r["eigs"]), grid, 0.06)
    dens /= dens.max()
    x0, x1 = 40, W - 40
    pts = [(x0 + (g - grid[0]) / (grid[-1] - grid[0]) * (x1 - x0), 92 - 64 * d) for g, d in zip(grid, dens)]
    fade = s.linear("footfade", [(0, "gold", 0), (0.3, "ink3", 0.8), (0.55, "gold", 1), (1, "gold", 0)])
    s.path(poly(simplify(pts, 0.1)), stroke=fade, sw=1.6)
    xm = math.log10(r["xmin"])
    tail = [(x, y) for (x, y), g in zip(pts, grid) if g >= xm]
    s.path(poly(simplify(tail, 0.1)), stroke="gold", sw=2, opacity=0.9, cls="breathe", attrs='style="--d:5s"')
    return s


# glyphs ---------------------------------------------------------------------
GLYPH_INK = "#8a93a3"


def glyphs():
    """Theme-neutral icons: mid-tone strokes that read on white and on #0d1117."""
    gold, teal, blue, green, ember = "#e8a82a", "#1fae9f", "#4f8df7", "#3fae5e", "#e2507a"
    ink = GLYPH_INK
    def svg(name, body, title):
        return name, (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="64" height="64" fill="none" '
                      f'role="img" aria-label="{title}" stroke-linecap="round" stroke-linejoin="round">{body}</svg>\n')
    out = [
        svg("start", f'<rect x="7" y="12" width="50" height="40" rx="7" stroke="{ink}" stroke-width="2.2"/>'
                     f'<path d="M17 27l7 6-7 6" stroke="{gold}" stroke-width="3"/><path d="M29 40h15" stroke="{ink}" stroke-width="3"/>', "Quick start"),
        svg("architecture", f'<rect x="6" y="26" width="14" height="12" rx="3" stroke="{ink}" stroke-width="2.2"/>'
                            f'<rect x="25" y="22" width="14" height="20" rx="3" fill="{gold}" stroke="{gold}" stroke-width="2.2"/>'
                            f'<rect x="44" y="12" width="14" height="14" rx="3" stroke="{blue}" stroke-width="2.2"/>'
                            f'<rect x="44" y="38" width="14" height="14" rx="3" stroke="{green}" stroke-width="2.2"/>'
                            f'<path d="M20 32h5M39 29l5-8M39 35l5 8" stroke="{ink}" stroke-width="2.2"/>', "Architecture"),
        svg("gpus", f'<rect x="16" y="16" width="32" height="32" rx="5" stroke="{ink}" stroke-width="2.2"/>'
                    f'<rect x="24" y="24" width="16" height="16" rx="2" fill="{teal}"/>'
                    + "".join(f'<path d="M{x} 9v7M{x} 48v7M9 {x}h7M48 {x}h7" stroke="{ink}" stroke-width="2.2"/>' for x in (24, 32, 40)), "GPUs"),
        svg("control", f'<path d="M12 16h40M12 32h40M12 48h40" stroke="{ink}" stroke-width="2.2"/>'
                       f'<circle cx="22" cy="16" r="5" fill="{gold}"/><circle cx="40" cy="32" r="5" fill="{blue}"/>'
                       f'<circle cx="28" cy="48" r="5" fill="{ember}"/>', "Runtime controls"),
        svg("spectrum", f'<path d="M8 54h48" stroke="{ink}" stroke-width="2.2"/>'
                        + "".join(f'<rect x="{10 + k * 6}" y="{54 - h}" width="4" height="{h}" rx="1" fill="{teal if k < 4 else gold}"/>'
                                  for k, h in enumerate((10, 26, 38, 30, 18, 11, 7, 4)))
                        + f'<path d="M10 44C18 10 26 12 34 34S48 50 56 51" stroke="{ink}" stroke-width="2"/>', "Spectrum"),
        svg("fingerprint", "".join(f'<rect x="{10 + (k % 4) * 11}" y="{10 + (k // 4) * 11}" width="9" height="9" rx="2" '
                                   f'fill="{(gold, teal, ink)[(k * 7 + k // 3) % 3]}" opacity="{(0.95, 0.55, 0.3, 0.8)[(k * 5) % 4]}"/>'
                                   for k in range(16)), "Fingerprints"),
        svg("zoo", f'<ellipse cx="32" cy="14" rx="20" ry="6" stroke="{ink}" stroke-width="2.2"/>'
                   f'<path d="M12 14v34c0 3.3 9 6 20 6s20-2.7 20-6V14M12 31c0 3.3 9 6 20 6s20-2.7 20-6" stroke="{ink}" stroke-width="2.2"/>'
                   f'<circle cx="24" cy="26" r="2.6" fill="{gold}"/><circle cx="38" cy="24" r="2.6" fill="{teal}"/>'
                   f'<circle cx="31" cy="44" r="2.6" fill="{blue}"/><circle cx="43" cy="42" r="2.6" fill="{gold}"/>', "Zoo"),
        svg("verify", f'<circle cx="32" cy="32" r="22" stroke="{ink}" stroke-width="2.2"/>'
                      f'<path d="M22 33l7 7 14-15" stroke="{green}" stroke-width="3.2"/>', "Verification"),
    ]
    for name, text in out:
        (OUT / f"glyph-{name}.svg").write_text(text)
        print(f"docs/assets/glyph-{name}.svg")


FIGURES = {"hero": hero, "measure": measure, "terminal": terminal, "architecture": architecture,
           "gpus": gpus, "lifecycle": lifecycle, "spectrum": spectrum, "fingerprint": fingerprint,
           "zoo": zoo, "footer": footer}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*")
    a = p.parse_args()
    for name, fn in FIGURES.items():
        if a.only and name not in a.only:
            continue
        for theme in ("dark", "light"):
            path = fn(theme).save(OUT / f"{name}-{theme}.svg")
            print(f"{path.relative_to(ROOT)}  {path.stat().st_size / 1024:.1f} KB")
    if not a.only or "glyphs" in a.only:
        glyphs()


if __name__ == "__main__":
    main()
