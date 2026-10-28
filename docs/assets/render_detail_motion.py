#!/usr/bin/env python3
"""Render two explanatory README loops from their editable SVG posters.

Requires Inkscape and Pillow. Motion is schematic, never runtime telemetry.
GPU pulses follow SVG centerlines; spectrum outlines follow tagged SVG shapes.
Only the two GIFs are retained. All intermediate files use a temporary directory.
"""

import math
import os
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw

from render_run_atlas import position, samples


ASSETS = Path(__file__).resolve().parent
WIDTH = 1000
FRAMES = 80
DURATION_MS = 100
ROUTES = (
    ("a-to-0", .02, .20, "#006ce4"),
    ("a-to-1", .23, .41, "#006ce4"),
    ("b-to-0", .10, .28, "#00693e"),
    ("b-to-1", .31, .49, "#00693e"),
    ("a-from-0", .45, .63, "#006ce4"),
    ("a-from-1", .66, .84, "#006ce4"),
    ("b-from-0", .53, .71, "#00693e"),
    ("b-from-1", .74, .92, "#00693e"),
)
COMPONENTS = ((4, "#e2aa28"), (3, "#006ce4"),
              (2, "#39967b"), (1, "#9272be"))


def rgba(hex_color, alpha):
    return tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5)) + (alpha,)


def gpu_overlay(draw, fraction, scale, elements):
    for name, start, end, color in ROUTES:
        progress = (fraction - start) / (end - start)
        if not 0 <= progress <= 1:
            continue
        # Quiet arrivals and departures; no full-panel flashing.
        alpha = round(230 * min(1, progress * 8, (1 - progress) * 8))
        for delay, radius, opacity in ((.12, 2, .3), (.06, 2.8, .5), (0, 4.4, 1)):
            if progress < delay:
                continue
            x, y = (p * scale for p in position(elements[name], progress - delay))
            r = radius * scale
            draw.ellipse((x - r, y - r, x + r, y + r), fill=rgba(color, round(alpha * opacity)))


def spectrum_overlay(draw, fraction, scale, elements):
    # Match one component of this diagonal example across all three panels.
    component, color = COMPONENTS[min(3, int(fraction * 4))]
    phase = (fraction * 4) % 1
    alpha = round(220 * math.sin(math.pi * phase) ** .7)
    for prefix in ("weight", "singular", "eigen"):
        element = elements[f"{prefix}-{component}"]
        if element.tag.endswith("rect"):
            x, y, width, height = (float(element.get(k)) for k in ("x", "y", "width", "height"))
            bounds = tuple(v * scale for v in (x - 4, y - 4, x + width + 4, y + height + 4))
            draw.rounded_rectangle(bounds, radius=7 * scale,
                                   outline=rgba(color, alpha), width=round(2.4 * scale))
        else:
            x, y = (float(element.get(k)) for k in ("cx", "cy"))
            bounds = tuple(v * scale for v in (x - 11, y - 11, x + 11, y + 11))
            draw.ellipse(bounds, outline=rgba(color, alpha), width=round(2.4 * scale))


def render(name, overlay_fn, scratch):
    svg_path = ASSETS / f"{name}.svg"
    svg = ET.parse(svg_path).getroot()
    elements = {node.get("id"): node for node in svg.iter() if node.get("id")}
    if name == "gpu-orchestration":
        elements = {route[0]: samples(elements[route[0]].get("d")) for route in ROUTES}
    png = scratch / f"{name}.png"
    environment = os.environ.copy()
    environment["INKSCAPE_PROFILE_DIR"] = str(scratch / "inkscape-profile")
    subprocess.run(["inkscape", str(svg_path), "--export-type=png",
                    f"--export-width={WIDTH * 2}", f"--export-filename={png}"],
                   env=environment, check=True, capture_output=True)
    base = Image.open(png).convert("RGBA")
    size = (WIDTH, round(base.height / 2))
    scale = base.width / 1200
    palette = base.convert("RGB").resize(size, Image.Resampling.LANCZOS).quantize(colors=248)
    transparent = base.getchannel("A").resize(size, Image.Resampling.LANCZOS).point(
        lambda alpha: 255 if alpha < 128 else 0)
    frames = []
    for index in range(FRAMES):
        overlay = Image.new("RGBA", base.size)
        overlay_fn(ImageDraw.Draw(overlay), index / FRAMES, scale, elements)
        frame = Image.alpha_composite(base, overlay).convert("RGB")
        frame = frame.resize(size, Image.Resampling.LANCZOS)
        indexed = frame.quantize(palette=palette, dither=Image.Dither.NONE)
        indexed.paste(255, mask=transparent)
        frames.append(indexed)
    output = ASSETS / f"{name}.gif"
    frames[0].save(output, save_all=True, append_images=frames[1:],
                   duration=DURATION_MS, loop=0, optimize=True, disposal=1,
                   transparency=255, background=255)
    print(f"{output.name}: {size[0]} × {size[1]}, {FRAMES * DURATION_MS} ms, "
          f"{output.stat().st_size:,} bytes")


def main():
    with tempfile.TemporaryDirectory(prefix="model-zoo-detail-motion-") as temporary:
        scratch = Path(temporary)
        render("gpu-orchestration", gpu_overlay, scratch)
        render("weight-to-spectrum", spectrum_overlay, scratch)


if __name__ == "__main__":
    main()
