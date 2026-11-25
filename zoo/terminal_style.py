"""Shared visual roles for the navigator, workflow pages and terminal progress."""

from dataclasses import dataclass
import os
import sys
import unicodedata


PALETTES = {
    "dark": dict(background=233, text=255, brand=81, accent=141, border=245,
                 muted=250, error=203, selection=24, glow=117, spark=222, aurora=209, rose=210),
    "light": dict(background=255, text=234, brand=25, accent=61, border=60,
                  muted=60, error=124, selection=189, glow=26, spark=94, aurora=22, rose=125),
}


@dataclass(frozen=True)
class PageLayout:
    """A shared content column with room for terminal autowrap and side gutters."""

    columns: int

    @property
    def gutter(self):
        usable = max(0, self.columns - 1)
        return max(2, (usable - 88) // 2) if usable >= 16 else 0

    @property
    def width(self):
        return min(88, max(0, self.columns - 1 - 2 * self.gutter))

    @property
    def indent(self):
        return " " * self.gutter


def cell_width(text):
    return sum(0 if unicodedata.combining(char) else
               2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
               for char in text)


def wrap_cells(text, width, *, literal=False):
    """Wrap guidance and paths without splitting a double-width cell."""
    if width < 1:
        return []
    rows = []
    for logical in text.split("\n"):
        while cell_width(logical) > width:
            cut = 0
            used = 0
            for char in logical:
                cells = cell_width(char)
                if used + cells > width:
                    break
                used += cells
                cut += 1
            if not cut:
                logical = "…" + logical[1:]
                continue
            if not literal:
                space = logical.rfind(" ", 1, cut + 1)
                if space > 0:
                    cut = space
            rows.append(logical[:cut])
            logical = logical[cut:] if literal else logical[cut:].lstrip()
        rows.append(logical)
    return rows


class Theme:
    def __init__(self, *, palette="auto", enabled=None, stream=None):
        stream = sys.stdout if stream is None else stream
        self.enabled = (stream.isatty() and "NO_COLOR" not in os.environ and os.getenv("TERM") != "dumb"
                        if enabled is None else enabled)
        self.palette = self.automatic_palette() if palette == "auto" else palette
        self.in_frame = False

    def backdrop(self):
        if not self.enabled or not self.in_frame:
            return ""
        colors = PALETTES[self.palette]
        return f"\x1b[38;5;{colors['text']};48;5;{colors['background']}m"

    @staticmethod
    def automatic_palette():
        try:
            background = int(os.environ.get("COLORFGBG", "").split(";")[-1])
        except ValueError:
            return "dark"
        return "light" if background in (7, 15) or 231 <= background <= 255 else "dark"

    def paint(self, value, role):
        if not self.enabled:
            return value
        weight = "1;" if role in ("brand", "heading") else ""
        color = PALETTES[self.palette]["brand" if role == "heading" else role]
        return f"\x1b[{weight}38;5;{color}m{value}\x1b[0m{self.backdrop()}"

    def brand(self, value):
        return self.paint(value, "brand")
