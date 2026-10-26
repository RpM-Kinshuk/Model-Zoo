#!/usr/bin/env python3
"""Render the README's flow animation from its editable SVG.

Requires Inkscape and Pillow. Draws schematic work pulses, not simulated metrics.
All intermediate PNGs live in a temporary directory; only run-atlas.gif is kept.
"""

import bisect
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw


ASSETS = Path(__file__).resolve().parent
WIDTH = 1000
FRAMES = 72
SVG_WIDTH = 1200
ROUTES = (
    ("manifest-flow", 0.00, 0.24, "#c79832"),
    ("worker-a-flow", 0.16, 0.46, "#006ce4"),
    ("worker-b-flow", 0.29, 0.59, "#00693e"),
    ("spectrum-a-flow", 0.43, 0.73, "#006ce4"),
    ("spectrum-b-flow", 0.56, 0.86, "#00693e"),
    ("release-flow", 0.73, 1.00, "#00693e"),
)


def samples(path):
    """Sample the atlas's absolute M/C/H paths, then parameterize by distance."""
    tokens = iter(re.findall(r"[MCH]|-?\d+(?:\.\d+)?", path))
    points = []
    current = (0.0, 0.0)
    for token in tokens:
        if token == "M":
            current = (float(next(tokens)), float(next(tokens)))
            points.append(current)
        elif token == "C":
            first = (float(next(tokens)), float(next(tokens)))
            second = (float(next(tokens)), float(next(tokens)))
            end = (float(next(tokens)), float(next(tokens)))
            for step in range(1, 161):
                t = step / 160
                a = 1 - t
                points.append(tuple(a**3 * current[i] + 3*a*a*t*first[i]
                                    + 3*a*t*t*second[i] + t**3*end[i] for i in (0, 1)))
            current = end
        elif token == "H":
            current = (float(next(tokens)), current[1])
            points.append(current)
        else:
            raise ValueError(f"Unsupported path token {token!r}")
    distances = [0.0]
    for previous, point in zip(points, points[1:]):
        distances.append(distances[-1] + math.dist(previous, point))
    return points, distances


def position(route, fraction):
    points, distances = route
    distance = min(1.0, max(0.0, fraction)) * distances[-1]
    i = min(bisect.bisect_right(distances, distance), len(points) - 1)
    length = distances[i] - distances[i - 1]
    t = (distance - distances[i - 1]) / length if length else 0
    return tuple(points[i - 1][k] + t * (points[i][k] - points[i - 1][k]) for k in (0, 1))


def main():
    svg = ET.parse(ASSETS / "run-atlas.svg").getroot()
    paths = {element.attrib["id"]: samples(element.attrib["d"])
             for element in svg.iter() if element.attrib.get("id") in {r[0] for r in ROUTES}}
    with tempfile.TemporaryDirectory(prefix="model-zoo-atlas-") as temporary:
        scratch = Path(temporary)
        png = scratch / "atlas.png"
        environment = os.environ.copy()
        environment["INKSCAPE_PROFILE_DIR"] = str(scratch / "inkscape-profile")
        subprocess.run(["inkscape", str(ASSETS / "run-atlas.svg"), "--export-type=png",
                        f"--export-width={WIDTH * 2}", f"--export-filename={png}"],
                       env=environment, check=True, capture_output=True)
        base = Image.open(png).convert("RGBA")
        height = round(base.height / 2)
        scale = base.width / SVG_WIDTH
        palette = base.convert("RGB").resize((WIDTH, height), Image.Resampling.LANCZOS).quantize(colors=248)
        frames = []
        for frame in range(FRAMES):
            time = frame / FRAMES
            overlay = Image.new("RGBA", base.size)
            draw = ImageDraw.Draw(overlay)
            for name, start, end, color in ROUTES:
                fraction = (time - start) / (end - start)
                if not 0 <= fraction <= 1:
                    continue
                rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
                for offset, radius, alpha in ((-0.045, 2, 45), (-0.022, 2.8, 100), (0, 4.3, 235)):
                    if fraction + offset < 0:
                        continue
                    x, y = (coordinate * scale for coordinate in position(paths[name], fraction + offset))
                    r = radius * scale
                    draw.ellipse((x-r, y-r, x+r, y+r), fill=(*rgb, alpha))
            composed = Image.alpha_composite(base, overlay).convert("RGB")
            composed = composed.resize((WIDTH, height), Image.Resampling.LANCZOS)
            frames.append(composed.quantize(palette=palette, dither=Image.Dither.NONE))
        output = ASSETS / "run-atlas.gif"
        frames[0].save(output, save_all=True, append_images=frames[1:], duration=100,
                       loop=0, optimize=True, disposal=1)
        print(f"{output.name}: {WIDTH} × {height}, {FRAMES} frames, {output.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
