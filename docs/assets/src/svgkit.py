"""SVG primitives for the README figures.

Every figure is drawn twice, once per GitHub theme, from the same code. Fonts are
subset to the glyphs a figure uses and embedded, because an SVG shown through
<img> cannot load external resources. Motion is CSS only, and every animated
element's base style is its resting state, so reduced motion shows a complete
still figure.
"""
from __future__ import annotations

import base64
import io
from collections import defaultdict
from functools import lru_cache
from html import escape
from pathlib import Path

from fontTools import subset
from fontTools.ttLib import TTFont

FONT_DIR = Path(__file__).with_name("fonts")

THEMES = {
    # a clear night: neutral ink, cool structure, warm light
    "dark": dict(
        page="#0d1117", panel="#0a0f16", panel2="#111925", lift="#182231",
        line="#243143", line2="#34445b", grid="#151e2b",
        ink="#f5f1e8", ink2="#c3cad6", ink3="#8b96a8",
        gold="#ffc457", gold_t="#ffd27c", ember="#ff6f8e", ember_t="#ff8fa6",
        teal="#3fdcc8", teal_t="#6ee7d6", sage="#a8dcb8",
        blue="#6aa8ff", blue_t="#8fbcff", green="#8be08f", green_t="#a6eaa8",
        red="#ff6b6b", red_t="#ff8a8a", violet="#b48cff", violet_t="#c6a6ff",
        magenta="#ff5ed8", tile="#16406b", tile_ink="#f5f9ff",
        neb1="#1fb7a6", neb2="#ff8a5c", neb3="#3d7bff", neb4="#c06cff", star="#fff4dc",
        glow=1.0, wash=0.55,
    ),
    # a star atlas on vellum: navy ink, sea green, gold leaf, rose
    "light": dict(
        page="#ffffff", panel="#fbf8f1", panel2="#fffdf9", lift="#f4efe5",
        line="#e3dccf", line2="#cec3b0", grid="#efe8db",
        ink="#16213a", ink2="#475266", ink3="#727c8f",
        gold="#cf8f12", gold_t="#8a5800", ember="#d9466f", ember_t="#a82850",
        teal="#0e9c8f", teal_t="#0a6c63", sage="#3f8f6a",
        blue="#2f6fdc", blue_t="#2152ad", green="#2b9a52", green_t="#1d703b",
        red="#cf3b3b", red_t="#a42828", violet="#7a4ddb", violet_t="#5a32b3",
        magenta="#c2369f", tile="#173a63", tile_ink="#fbf8f1",
        neb1="#5fd3c4", neb2="#ffb08a", neb3="#8fb6ff", neb4="#d5b3ff", star="#4f5b78",
        glow=0.55, wash=0.8,
    ),
}

# role: (css family, file, weight, style)
ROLES = {
    "sans": ("MZ Text", "mz-text-regular", 400, "normal"),
    "sansb": ("MZ Text", "mz-text-bold", 600, "normal"),
    "texti": ("MZ Text", "mz-text-italic", 400, "italic"),
    "serif": ("MZ Serif", "mz-serif-regular", 400, "normal"),
    "serifi": ("MZ Serif", "mz-serif-italic", 400, "italic"),
    "mono": ("MZ Mono", "mz-mono-regular", 400, "normal"),
    "math": ("MZ Math", "mz-math-italic", 400, "italic"),
    "mathr": ("MZ Math", "mz-math-regular", 400, "normal"),
    "op": ("MZ Op", "mz-math-ops", 400, "normal"),
}
# Glyphs missing from a role's font are taken from these, in order.
FALLBACK = {
    "sans": ["sansb", "mathr", "op"], "sansb": ["sans", "mathr", "op"], "texti": ["math", "sans", "op"],
    "serif": ["serifi", "sans", "op"], "serifi": ["serif", "texti", "op"],
    "mono": ["sans", "op"], "math": ["mathr", "op", "sans"],
    "mathr": ["math", "op", "sans"], "op": ["mathr", "sans"],
}
STACKS = {
    "MZ Text": "'MZ Text','MZ Math','MZ Op','STIX Two Text',Georgia,'Times New Roman',serif",
    "MZ Serif": "'MZ Serif','Instrument Serif','MZ Text',Georgia,'Times New Roman',serif",
    "MZ Mono": "'MZ Mono','DM Mono','MZ Text','MZ Op',ui-monospace,SFMono-Regular,Menlo,Consolas,monospace",
    "MZ Math": "'MZ Math','MZ Op','MZ Text','STIX Two Text','Cambria Math','Times New Roman',serif",
    "MZ Op": "'MZ Op','MZ Math','MZ Text','STIX Two Math','Cambria Math',serif",
}

# Keyframes use fixed values; per-element timing comes from --d and --t, which
# are only read outside @keyframes, where every engine resolves them.
BASE_CSS = """
text{font-kerning:normal;font-variant-ligatures:common-ligatures;white-space:pre}
.comet,.comet-head{fill:none;stroke-linecap:round;opacity:0;animation:comet var(--d,3.2s) linear infinite;animation-delay:var(--t,0s)}
.comet{stroke-dasharray:16 200;stroke-dashoffset:16}
.comet-head{stroke-dasharray:2.4 200;stroke-dashoffset:2.4;animation-name:comet-head}
@keyframes comet{0%{stroke-dashoffset:16;opacity:0}6%{opacity:1}86%{opacity:1}100%{stroke-dashoffset:-100;opacity:0}}
@keyframes comet-head{0%{stroke-dashoffset:2.4;opacity:0}6%{opacity:1}86%{opacity:1}100%{stroke-dashoffset:-113.6;opacity:0}}
@keyframes breathe{0%,100%{opacity:.35}50%{opacity:1}}
@keyframes draw{from{stroke-dashoffset:100}to{stroke-dashoffset:0}}
@keyframes fade{from{opacity:0}to{opacity:1}}
.breathe{animation:breathe var(--d,4s) ease-in-out infinite;animation-delay:var(--t,0s)}
@keyframes twinkle{0%,100%{opacity:.15}50%{opacity:1}}
.tw{animation:twinkle var(--d,3s) ease-in-out infinite;animation-delay:var(--t,0s)}
@keyframes drift{0%,100%{transform:translate(0,0)}50%{transform:translate(14px,-8px)}}
.drift{animation:drift var(--d,18s) ease-in-out infinite;animation-delay:var(--t,0s)}
@media (prefers-reduced-motion:reduce){*{animation:none!important}}
"""


@lru_cache(maxsize=None)
def _font(file: str) -> TTFont:
    return TTFont(FONT_DIR / f"{file}.woff2")


@lru_cache(maxsize=None)
def _cmap(file: str) -> dict:
    return _font(file).getBestCmap()


def has_glyph(role: str, ch: str) -> bool:
    return ord(ch) in _cmap(ROLES[role][1])


def advance(role: str, text: str, size: float) -> float:
    """Approximate rendered width without kerning; good enough for layout."""
    width = 0.0
    for ch in text:
        for r in [role] + FALLBACK[role]:
            file = ROLES[r][1]
            if ord(ch) in _cmap(file):
                font = _font(file)
                glyph = _cmap(file)[ord(ch)]
                width += font["hmtx"][glyph][0] / font["head"].unitsPerEm * size
                break
        else:
            width += 0.5 * size
    return width


def fmt(v: float) -> str:
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return "0" if s == "-0" else s


class Svg:
    def __init__(self, w: int, h: int, theme: str, title: str, desc: str):
        self.w, self.h, self.theme = w, h, theme
        self.c = THEMES[theme]
        self.title, self.desc = title, desc
        self.defs: list[str] = []
        self.css: list[str] = []
        self.body: list[str] = []
        self.glyphs: dict[str, set[str]] = defaultdict(set)
        self._ids: set[str] = set()

    # colors ----------------------------------------------------------------
    def col(self, v):
        if v is None:
            return None
        return self.c.get(v, v)

    # bookkeeping -----------------------------------------------------------
    def add(self, *parts: str):
        self.body.extend(parts)

    def define(self, key: str, markup: str):
        if key not in self._ids:
            self._ids.add(key)
            self.defs.append(markup)

    def style(self, css: str):
        self.css.append(css)

    def _use(self, role: str, text: str):
        for ch in text:
            if ch in " \n":
                continue
            for r in [role] + FALLBACK[role]:
                if has_glyph(r, ch):
                    self.glyphs[r].add(ch)
                    break

    # text ------------------------------------------------------------------
    def text(self, x, y, content, role="sans", size=16, fill="ink", anchor=None,
             cls=None, ls=None, opacity=None, attrs="", weight=None):
        """content: a string, or a list of runs (text, {role, size, fill, sup, sub, ls})."""
        runs = [(content, {})] if isinstance(content, str) else content
        out = []
        shift = 0.0
        for text, opt in runs:
            r = opt.get("role", role)
            self._use(r, text)
            fam, _, wt, st = ROLES[r]
            sz = opt.get("size", size * (0.68 if opt.get("sup") or opt.get("sub") else 1))
            target = -size * 0.38 if opt.get("sup") else size * 0.2 if opt.get("sub") else 0.0
            dy = target - shift
            shift = target
            a = [f'font-family="{fam}"' if r != role else "", ]
            if r != role:
                a.append(f'font-weight="{wt}"' if wt != 400 else "")
                a.append(f'font-style="{st}"' if st != "normal" else "")
            if sz != size:
                a.append(f'font-size="{fmt(sz)}"')
            if dy:
                a.append(f'dy="{fmt(dy)}"')
            if opt.get("fill"):
                a.append(f'fill="{self.col(opt["fill"])}"')
            if opt.get("ls") is not None:
                a.append(f'letter-spacing="{opt["ls"]}"')
            if opt.get("dx"):
                a.append(f'dx="{fmt(opt["dx"])}"')
            if opt.get("cls"):
                a.append(f'class="{opt["cls"]}"')
            attr = " ".join(p for p in a if p)
            out.append(f"<tspan {attr}>{escape(text)}</tspan>" if attr else escape(text))
        fam, _, wt, st = ROLES[role]
        a = [f'x="{fmt(x)}" y="{fmt(y)}" font-family="{fam}" font-size="{fmt(size)}" fill="{self.col(fill)}"']
        if (weight or wt) != 400:
            a.append(f'font-weight="{weight or wt}"')
        if st != "normal":
            a.append(f'font-style="{st}"')
        if anchor:
            a.append(f'text-anchor="{anchor}"')
        if ls is not None:
            a.append(f'letter-spacing="{ls}"')
        if opacity is not None:
            a.append(f'opacity="{opacity}"')
        if cls:
            a.append(f'class="{cls}"')
        if attrs:
            a.append(attrs)
        s = f"<text {' '.join(a)}>{''.join(out)}</text>"
        self.add(s)
        return s

    @staticmethod
    def width(content, role="sans", size=16) -> float:
        runs = [(content, {})] if isinstance(content, str) else content
        total = 0.0
        for text, opt in runs:
            r = opt.get("role", role)
            sz = opt.get("size", size * (0.68 if opt.get("sup") or opt.get("sub") else 1))
            total += advance(r, text, sz)
        return total

    # shapes ----------------------------------------------------------------
    def rect(self, x, y, w, h, rx=0, fill=None, stroke=None, sw=1, opacity=None, cls=None, attrs=""):
        a = [f'x="{fmt(x)}" y="{fmt(y)}" width="{fmt(w)}" height="{fmt(h)}"']
        if rx:
            a.append(f'rx="{fmt(rx)}"')
        a.append(f'fill="{self.col(fill) or "none"}"')
        if stroke:
            a.append(f'stroke="{self.col(stroke)}" stroke-width="{fmt(sw)}"')
        if opacity is not None:
            a.append(f'opacity="{opacity}"')
        if cls:
            a.append(f'class="{cls}"')
        if attrs:
            a.append(attrs)
        s = f"<rect {' '.join(a)}/>"
        self.add(s)
        return s

    def line(self, x1, y1, x2, y2, stroke="line", sw=1, dash=None, opacity=None, cls=None, attrs=""):
        a = [f'x1="{fmt(x1)}" y1="{fmt(y1)}" x2="{fmt(x2)}" y2="{fmt(y2)}" stroke="{self.col(stroke)}" stroke-width="{fmt(sw)}"']
        if dash:
            a.append(f'stroke-dasharray="{dash}"')
        if opacity is not None:
            a.append(f'opacity="{opacity}"')
        if cls:
            a.append(f'class="{cls}"')
        if attrs:
            a.append(attrs)
        s = f"<line {' '.join(a)}/>"
        self.add(s)
        return s

    def path(self, d, stroke=None, sw=1, fill=None, dash=None, opacity=None, cls=None, attrs="", cap="round", join="round"):
        a = [f'd="{d}"', f'fill="{self.col(fill) or "none"}"']
        if stroke:
            a.append(f'stroke="{self.col(stroke)}" stroke-width="{fmt(sw)}" stroke-linecap="{cap}" stroke-linejoin="{join}"')
        if dash:
            a.append(f'stroke-dasharray="{dash}"')
        if opacity is not None:
            a.append(f'opacity="{opacity}"')
        if cls:
            a.append(f'class="{cls}"')
        if attrs:
            a.append(attrs)
        s = f"<path {' '.join(a)}/>"
        self.add(s)
        return s

    def circle(self, cx, cy, r, fill=None, stroke=None, sw=1, opacity=None, cls=None, attrs=""):
        a = [f'cx="{fmt(cx)}" cy="{fmt(cy)}" r="{fmt(r)}" fill="{self.col(fill) or "none"}"']
        if stroke:
            a.append(f'stroke="{self.col(stroke)}" stroke-width="{fmt(sw)}"')
        if opacity is not None:
            a.append(f'opacity="{opacity}"')
        if cls:
            a.append(f'class="{cls}"')
        if attrs:
            a.append(attrs)
        s = f"<circle {' '.join(a)}/>"
        self.add(s)
        return s

    def open(self, attrs=""):
        self.add(f"<g {attrs}>" if attrs else "<g>")

    def close(self):
        self.add("</g>")

    # effects ---------------------------------------------------------------
    def glow_filter(self, key="glow", blur=6, strength=1.0):
        self.define(key, (
            f'<filter id="{key}" x="-50%" y="-50%" width="200%" height="200%" color-interpolation-filters="sRGB">'
            f'<feGaussianBlur stdDeviation="{blur}" result="b"/>'
            f'<feComponentTransfer in="b" result="g"><feFuncA type="linear" slope="{strength}"/></feComponentTransfer>'
            '<feMerge><feMergeNode in="g"/><feMergeNode in="SourceGraphic"/></feMerge></filter>'))
        return f"url(#{key})"

    def blur_filter(self, key, blur):
        self.define(key, f'<filter id="{key}" x="-60%" y="-60%" width="220%" height="220%"><feGaussianBlur stdDeviation="{blur}"/></filter>')
        return f"url(#{key})"

    def radial(self, key, color, opacity=1.0, stop=1.0):
        self.define(key, (
            f'<radialGradient id="{key}"><stop offset="0" stop-color="{self.col(color)}" stop-opacity="{opacity}"/>'
            f'<stop offset="{stop}" stop-color="{self.col(color)}" stop-opacity="0"/></radialGradient>'))
        return f"url(#{key})"

    def linear(self, key, stops, x1=0, y1=0, x2=1, y2=0, units=None):
        s = "".join(f'<stop offset="{o}" stop-color="{self.col(c)}" stop-opacity="{op}"/>' for o, c, op in stops)
        u = f' gradientUnits="{units}"' if units else ""
        self.define(key, f'<linearGradient id="{key}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"{u}>{s}</linearGradient>')
        return f"url(#{key})"

    def comet(self, d, color="gold", dur=3.2, delay=0.0, sw=2.2, opacity=0.55, glow=True):
        """A light that travels along d: a soft tail and a bright head. Hidden when still."""
        flt = f' filter="{self.glow_filter("cometglow", 3, 1.2 * self.c["glow"])}"' if glow else ""
        style = f'style="--d:{fmt(dur)}s;--t:{fmt(delay)}s"'
        col = self.col(color)
        head = self.c["ink"] if self.theme == "dark" else col
        self.add(
            f'<g{flt}><path d="{d}" pathLength="100" class="comet" stroke="{col}" stroke-opacity="{opacity}" '
            f'stroke-width="{fmt(sw)}" {style}/>'
            f'<path d="{d}" pathLength="100" class="comet-head" stroke="{head}" stroke-width="{fmt(sw * 1.5)}" {style}/></g>')

    def arrowhead(self, x, y, angle=0, size=7, fill="ink3"):
        import math
        a = math.radians(angle)
        pts = [(x, y),
               (x - size * math.cos(a) + size * 0.55 * math.sin(a), y - size * math.sin(a) - size * 0.55 * math.cos(a)),
               (x - size * math.cos(a) - size * 0.55 * math.sin(a), y - size * math.sin(a) + size * 0.55 * math.cos(a))]
        self.path("M" + " L".join(f"{fmt(px)} {fmt(py)}" for px, py in pts) + " Z", fill=fill)

    # output ----------------------------------------------------------------
    def _font_faces(self) -> str:
        faces = []
        for role, chars in sorted(self.glyphs.items()):
            if not chars:
                continue
            fam, file, wt, st = ROLES[role]
            font = TTFont(FONT_DIR / f"{file}.woff2")
            opts = subset.Options()
            opts.flavor = "woff2"
            opts.layout_features = ["kern", "liga", "calt", "ccmp", "mark", "mkmk"]
            opts.name_IDs = [0, 1, 2, 4, 6, 13, 14]
            opts.notdef_outline = False
            opts.hinting = False
            sub = subset.Subsetter(opts)
            sub.populate(text="".join(sorted(chars)))
            sub.subset(font)
            buf = io.BytesIO()
            font.flavor = "woff2"
            font.save(buf)
            data = base64.b64encode(buf.getvalue()).decode()
            faces.append(f"@font-face{{font-family:'{fam}';font-weight:{wt};font-style:{st};"
                         f"src:url(data:font/woff2;base64,{data}) format('woff2')}}")
        return "".join(faces)

    def render(self) -> str:
        families = sorted({ROLES[r][0] for r in self.glyphs})
        stacks = "".join(f"[font-family=\"{f}\"]{{font-family:{STACKS[f]}}}" for f in families)
        css = self._font_faces() + stacks + BASE_CSS + "\n".join(self.css)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}" '
            f'role="img" aria-labelledby="title desc" fill="none" xml:space="preserve">'
            f'<title id="title">{escape(self.title)}</title><desc id="desc">{escape(self.desc)}</desc>'
            f"<style>{css}</style><defs>{''.join(self.defs)}</defs>"
            + "".join(self.body) + "</svg>\n")

    def save(self, path: Path):
        path.write_text(self.render(), encoding="utf-8")
        return path
