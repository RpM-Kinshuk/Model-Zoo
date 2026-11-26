"""Responsive, single-owner terminal navigator; workflow execution lives in tui.

The frame builder is pure. Only the event loop touches the terminal, and curses
restores its modes before any subprocess or ordinary line prompt runs.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import time
import math
import random
from secrets import randbits
from functools import lru_cache
import unicodedata
from typing import Sequence

from zoo.terminal_style import PALETTES, PageLayout


ANIMATION_FPS = 12
ATLAS_TAGLINE = "A living atlas of evidence"
DISCOVERY_TAGLINE = "Collect. Measure. Discover."


class MotionClock:
    def __init__(self):
        self.elapsed = 0.0
        self.started = None

    def set_running(self, running: bool) -> None:
        if running and self.started is None:
            self.started = time.monotonic()
        elif not running and self.started is not None:
            self.elapsed += time.monotonic() - self.started
            self.started = None

    def tick(self) -> int:
        current = 0 if self.started is None else time.monotonic() - self.started
        return int((self.elapsed + current) * ANIMATION_FPS)


@dataclass(frozen=True)
class Workflow:
    key: str
    title: str
    description: str
    group: str


WORKFLOWS = (
    Workflow('1', 'Prepare a collection', 'Resolve model versions and check which entries can run', 'COLLECT'),
    Workflow('2', 'Measure pinned models', 'Check GPUs and review the run before downloading weights', 'COLLECT'),
    Workflow('3', 'Summarize a run', 'Refresh the run summary from existing measurements', 'COLLECT'),
    Workflow('4', 'Publish completed evidence', 'Create immutable local model packages', 'ASSEMBLE'),
    Workflow('5', 'Build / refresh the zoo', 'Turn model packages into an offline query snapshot', 'ASSEMBLE'),
    Workflow('6', 'Measure model deltas', 'Compare detected lineage candidates', 'ASSEMBLE'),
    Workflow('7', 'Explore the zoo', 'Search metadata and compare weight fingerprints', 'DISCOVER'),
    Workflow('8', 'Scan infrastructure', 'Inspect GPUs, RAM, disk and Hub availability', 'OPERATE'),
)


@dataclass(frozen=True)
class Line:
    text: str
    role: str = 'text'
    artwork_start: int | None = None
    highlights: tuple[tuple[int, str], ...] = ()


@lru_cache(maxsize=1)
def spiral_galaxy() -> tuple[str, ...]:
    """Draw a face-on spiral at Braille subcell resolution (66 by 36 dots).

    A compact bulge and two broad logarithmic arms leave dark inter-arm lanes.
    The fixed dust pattern gives texture without introducing animation noise.
    """
    cells = [[0] * 33 for _ in range(9)]
    bits = ((0, 3), (1, 4), (2, 5), (6, 7))
    for py in range(36):
        for px in range(66):
            x, y = (px - 32.5) / 30, (py - 17.5) / 16
            radius = math.hypot(x, y)
            angle = math.atan2(y, x)
            arm_distance = math.sin(angle - 0.9 - 3.2 * math.log(radius + 0.12))
            arms = math.exp(-arm_distance ** 2 / 0.16) * math.exp(-radius * 1.5)
            bulge = math.exp(-radius ** 2 / 0.018)
            density = min(1, 2.2 * arms + bulge + 0.12 * math.exp(-radius * radius / 0.5))
            dust = ((px * 73 + py * 151 + px * py * 19) % 257) / 257
            if radius < 1 and density > 0.12 and dust < density:
                cells[py // 4][px // 2] |= 1 << bits[py % 4][px % 2]
    artwork = [''.join(chr(0x2800 + bits) if bits else ' ' for bits in row) for row in cells]
    artwork[4] = artwork[4][:16] + '✦' + artwork[4][17:]
    return tuple(artwork)


def _star_cycle(seed: int, tick: int) -> tuple[int, int, int]:
    period = ANIMATION_FPS * 18 + seed % (ANIMATION_FPS * 12 + 1)
    generation, age = divmod(tick + (seed >> 8) % period, period)
    return generation, age, period * 4 // 5


def star_brightness(seed: int, tick: int) -> int:
    """Fade in and out, then stay dark before moving to a new position."""
    _, age, lit_ticks = _star_cycle(seed, tick)
    if age >= lit_ticks:
        return 0
    return round(63 * math.sin(math.pi * age / lit_ticks) ** 4)


@lru_cache(maxsize=384)
def star_rgb(light: bool, family: int, level: int) -> tuple[int, int, int]:
    background = (238, 238, 238) if light else (18, 18, 18)
    targets = ((24, 75, 145), (145, 55, 20), (100, 45, 145)) if light else (
        (220, 239, 255), (255, 190, 120), (185, 155, 255))
    return tuple(round(a + (b - a) * level / 63)
                 for a, b in zip(background, targets[family]))


@lru_cache(maxsize=384)
def star_color(light: bool, family: int, level: int) -> int:
    """Find the closest shade on a 256-color terminal."""
    target = star_rgb(light, family, level)
    steps = (0, 95, 135, 175, 215, 255)
    candidates = [(16 + 36*r + 6*g + b, (red, green, blue))
                  for r, red in enumerate(steps) for g, green in enumerate(steps)
                  for b, blue in enumerate(steps)]
    candidates += [(232 + index, (8 + index*10,) * 3) for index in range(24)]
    return min(candidates, key=lambda item: sum((a-b)**2 for a, b in zip(item[1], target)))[0]


@lru_cache(maxsize=128)
def _star_sites(bounds: tuple[tuple[int, int], ...], field_seed: int) -> tuple[tuple[int, int], ...]:
    candidates = [(row, column) for row, (start, stop) in enumerate(bounds)
                  for column in range(start, stop)]
    random.Random(field_seed).shuffle(candidates)
    sites = []
    blocked = set()
    for row, column in candidates:
        if (row, column) in blocked:
            continue
        sites.append((row, column))
        # Terminal cells are taller than they are wide. Reserve more space
        # across each row than vertically.
        blocked.update((row + dy, column + dx) for dy in (-1, 0, 1) for dx in range(-3, 4))
    return tuple(sites)


@lru_cache(maxsize=128)
def _star_seeds(field_seed: int, count: int) -> tuple[int, ...]:
    generator = random.Random(field_seed)
    return tuple(generator.getrandbits(64) for _ in range(count))


@lru_cache(maxsize=512)
def _star_cell_index(seed: int, generation: int, count: int) -> int:
    return random.Random(f"{seed}:{generation}").randrange(count)


def _scatter_stars(lines: list[Line], tick: int, field_seed: int) -> list[Line]:
    bounds = []
    for line in lines:
        end = line.artwork_start if line.artwork_start is not None else len(line.text)
        start = len(line.text[:end].rstrip())
        bounds.append((start + 3, end - 3))
    sites = _star_sites(tuple(bounds), field_seed)
    if not sites:
        return lines

    area = sum(max(0, stop - start) for start, stop in bounds)
    count = min(max(1, len(sites) // 2), max(2, area // 32))
    seeds = _star_seeds(field_seed, count)
    # Give each star its own set of random, separated positions. A new fade
    # cannot crowd a star that is already visible.
    visible = []
    for index, seed in enumerate(seeds):
        level = star_brightness(seed, tick)
        if not level:
            continue
        generation, _, _ = _star_cycle(seed, tick)
        positions = sites[index::len(seeds)]
        cell = positions[_star_cell_index(seed, generation, len(positions))]
        visible.append((cell, level, index % 2))

    text = [list(line.text) for line in lines]
    highlights: list[list[tuple[int, str]]] = [[] for _ in lines]
    for (row, column), level, family in sorted(visible):
        dot = (row * 31 + column + field_seed) % 8
        text[row][column] = chr(0x2800 + (1 << dot))
        highlights[row].append((column, f"dust{family}_{level}"))
    return [replace(line, text="".join(text[row]), highlights=tuple(highlights[row]))
            if highlights[row] else line for row, line in enumerate(lines)]


def artwork_role(offset: int, width: int) -> str:
    spectrum = ("glow", "accent", "rose", "spark", "aurora")
    distance = abs(offset - width // 2)
    return "spark" if distance < 3 else spectrum[min(4, distance // 4)]


def brand_header(width: int, height: int, tick: int = 0, *,
                 max_rows: int | None = None, star_seed: int = 0) -> list[Line]:
    """Fit the same galaxy and title to the available space on each page."""
    if width < 25 or height < 10:
        return [Line("MODEL ZOO"[:width], "brand")]
    budget = height if max_rows is None else max_rows
    if height >= 34 and width >= 72 and budget >= 11:
        art_width, art_height = 33, 9
    elif height >= 22 and width >= 52 and budget >= 7:
        art_width = 33 if width - 33 >= len(DISCOVERY_TAGLINE) + 2 else 17
        art_height = 5
    elif height >= 16 and width >= 36 and budget >= 4:
        art_width, art_height = 17, 3
    else:
        art_width, art_height = 9, 1

    first_row = (9 - art_height) // 2
    first_column = (33 - art_width) // 2
    artwork_start = width - art_width
    rows = []
    if height >= 16 and budget > art_height:
        rows.append(Line(" " * width, "stars"))
    title_row = min(art_height // 2, max(0, art_height - 3))
    for index, row in enumerate(spiral_galaxy()[first_row:first_row + art_height]):
        title = "MODEL ZOO" if index == title_row else ""
        if index == title_row + 1 and artwork_start >= len(ATLAS_TAGLINE) + 2:
            title = ATLAS_TAGLINE
        if index == title_row + 2 and artwork_start >= len(DISCOVERY_TAGLINE) + 2:
            title = DISCOVERY_TAGLINE
        glyphs = row[first_column:first_column + art_width]
        rows.append(Line(title.ljust(artwork_start) + glyphs, "brand" if index == title_row else "muted" if title else "text",
                         artwork_start=artwork_start))
    if height >= 22 and len(rows) < budget:
        rows.append(Line(""))
    return _scatter_stars(rows, tick, star_seed)


def matching_workflows(query: str) -> list[Workflow]:
    words = query.casefold().split()
    return [item for item in WORKFLOWS
            if all(word in f'{item.title} {item.description} {item.group}'.casefold()
                   for word in words)]


def _fit_cells(value: str, width: int) -> str:
    """Clip terminal cells and discard control characters in workspace paths."""
    result, used = [], 0
    for character in value:
        if unicodedata.category(character).startswith('C'):
            continue
        cells = (0 if unicodedata.combining(character) else
                 2 if unicodedata.east_asian_width(character) in {'W', 'F'} else 1)
        if used + cells > width:
            break
        result.append(character)
        used += cells
    return ''.join(result)


def compose_frame(width: int, height: int, selected: int, query: str,
                  status: Sequence[str], tick: int, *, help_open: bool = False,
                  searching: bool = False, motion: bool = False, star_seed: int = 0) -> list[Line]:
    """Budget rows before drawing; reserve the last column to avoid autowrap."""
    layout = PageLayout(width)
    width, height = layout.width, max(0, height)
    if not height:
        return []
    items = matching_workflows(query)
    selected = min(max(0, selected), max(0, len(items) - 1))
    if height < 5:
        title = items[selected].title if items else 'No matches'
        rows = [Line('MODEL ZOO', 'brand'), Line(title, 'selected'), Line('Resize for more · q quit', 'muted')]
        return [Line(_fit_cells(row.text, width), row.role) for row in rows[:height]]
    rows = brand_header(width, height, tick, star_seed=star_seed)
    if height >= 18 and status:
        rows += [Line(line, 'muted') for line in status[:2]]
        rows.append(Line(''))
    rows += [Line('Search  ' + query + ('_' if searching else '')
                  if query or searching else 'Workflows', 'accent')]
    if width >= 48:
        footer = [Line('↑↓ Select · Enter · / Search · Esc Quit', 'muted'),
                  Line(f'{selected + 1 if items else 0}/{len(items)}  ·  T Theme  ·  '
                       f'M Motion {"on" if motion else "off"}  ·  ? Help', 'muted')]
    elif width >= 30:
        footer = [Line('↑/↓ Select  ·  Enter Open', 'muted'), Line('/ Search  ·  Esc Quit', 'muted')]
    else:
        footer = [Line('↑↓ Enter / Q', 'muted')]
    if searching:
        footer = [Line('Type to filter  ·  Enter Open', 'accent'), Line('Esc Clear search', 'muted')]
    elif help_open:
        footer = [Line('1–8 Open directly  ·  Esc Close help', 'accent'),
                  Line('Tab Next  ·  T Theme  ·  M Motion  ·  ? Close help', 'muted')]
    budget = max(0, height - len(rows) - len(footer))
    detailed = width >= 65 and budget >= len(items) * 2
    spacious = budget >= len(items) * (3 if detailed else 2)
    stride = (2 if detailed else 1) + int(spacious)
    count = max(1, budget // stride)
    start = max(0, min(selected - count // 2, len(items) - count))
    if not items:
        rows += [Line('  No matches. Esc clears search.', 'muted')][:budget]
    else:
        for index in range(start, min(len(items), start + count)):
            item = items[index]
            marker = '✦' if index == selected else '·'
            rows += [Line(f'{marker}  {item.key}  {item.title}', 'selected' if index == selected else
                              {'1': 'glow', '2': 'spark', '3': 'aurora', '4': 'rose',
                               '5': 'accent', '6': 'spark', '7': 'glow', '8': 'rose'}[item.key])]
            if detailed:
                rows += [Line('      ' + item.description, 'muted')]
            if spacious:
                rows.append(Line(''))
    padding = max(0, height - len(footer) - len(rows))
    if width >= 50 and height >= 22:
        rows += _scatter_stars([Line(" " * width, "stars")] * padding, tick, star_seed ^ 1)
    else:
        rows += [Line("")] * padding
    rows += footer
    margin = layout.indent
    return [Line(margin + _fit_cells(row.text, width), row.role,
                 row.artwork_start + len(margin) if row.artwork_start is not None else None,
                 tuple((column + len(margin), role) for column, role in row.highlights
                       if column < width))
            for row in rows[:height]]


def navigate(status: Sequence[str], *, palette: str = 'dark', motion: bool = False,
             color: bool = True, star_seed: int | None = None, clock: MotionClock | None = None) -> tuple[str, str, bool]:
    """Run an alternate-screen session and return a workflow and preferences."""
    import curses

    clock = clock or MotionClock()
    clock.set_running(motion)
    if star_seed is None:
        star_seed = randbits(64)
    if hasattr(curses, "set_escdelay"):
        curses.set_escdelay(75)

    def session(screen):
        nonlocal palette, motion
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        screen.keypad(True)
        screen.timeout(round(1000 / ANIMATION_FPS) if motion else -1)
        selected, query, searching, help_open = 0, '', False, False
        last_frame = None
        colors_available = color and curses.has_colors()
        if colors_available:
            curses.start_color()
            try:
                curses.use_default_colors()
            except curses.error:
                pass

        def attributes():
            roles = {'text': curses.A_NORMAL, 'brand': curses.A_BOLD,
                     'accent': curses.A_BOLD, 'muted': curses.A_NORMAL,
                     'galaxy': curses.A_NORMAL, 'aurora': curses.A_BOLD, 'rose': curses.A_BOLD,
                     'stars': curses.A_NORMAL, 'glow': curses.A_BOLD, 'spark': curses.A_BOLD, 'selected': curses.A_REVERSE | curses.A_BOLD}
            if colors_available:
                # Explicit background makes contrast independent of emulator theme.
                light = palette == 'light'
                background = curses.COLOR_WHITE if light else curses.COLOR_BLACK
                foreground = curses.COLOR_BLACK if light else curses.COLOR_WHITE
                accent = curses.COLOR_BLUE if light else curses.COLOR_CYAN
                secondary, muted, selection = curses.COLOR_MAGENTA, foreground, accent
                glow, spark = ((curses.COLOR_BLUE, curses.COLOR_RED) if light else
                               (curses.COLOR_CYAN, curses.COLOR_YELLOW))
                aurora, rose = (curses.COLOR_RED if light else curses.COLOR_GREEN), curses.COLOR_MAGENTA
                selected_foreground = background
                if curses.COLORS >= 256:
                    colors = PALETTES[palette]
                    background, foreground, accent, secondary, muted, selection = (
                        colors["background"], colors["text"], colors["brand"], colors["accent"],
                        colors["muted"], colors["selection"])
                    glow, spark, aurora, rose = (colors[key] for key in ("glow", "spark", "aurora", "rose"))
                    selected_foreground = foreground
                for index, (fg, bg) in enumerate(((foreground, background), (accent, background),
                                                 (secondary, background), (selected_foreground, selection),
                                                 (muted, background), (glow, background),
                                                 (spark, background), (aurora, background), (rose, background)), 1):
                    curses.init_pair(index, fg, bg)
                roles.update(text=curses.color_pair(1), muted=curses.color_pair(5),
                             brand=curses.color_pair(2) | curses.A_BOLD,
                             accent=curses.color_pair(3) | curses.A_BOLD,
                             galaxy=curses.color_pair(3),
                             aurora=curses.color_pair(8) | curses.A_BOLD,
                             rose=curses.color_pair(9) | curses.A_BOLD,
                             glow=curses.color_pair(6) | curses.A_BOLD,
                             spark=curses.color_pair(7) | curses.A_BOLD,
                             stars=curses.color_pair(3), selected=curses.color_pair(4) | curses.A_BOLD)
                screen.bkgd(' ', roles['text'])
            for family in range(3):
                for level in range(64):
                    name = f'star{level}' if family == 0 else f'dust{family - 1}_{level}'
                    roles[name] = curses.A_DIM if level < 28 else (curses.A_BOLD if level >= 48 else curses.A_NORMAL)
                    if colors_available:
                        if curses.COLORS >= 256 and curses.COLOR_PAIRS >= 202:
                            pair = 10 + family * 64 + level
                            shade = star_color(palette == 'light', family, level)
                            curses.init_pair(pair, shade, background)
                            roles[name] = curses.color_pair(pair)
                        else:
                            fallback = 'muted' if level < 28 else 'accent' if level < 48 else 'spark'
                            roles[name] |= roles[fallback]
            return roles

        roles = attributes()
        while True:
            height, width = screen.getmaxyx()
            tick = clock.tick()
            frame = compose_frame(width, height, selected, query, status, tick,
                                  help_open=help_open, searching=searching, motion=motion, star_seed=star_seed)
            if frame != last_frame:
                previous = last_frame if last_frame is not None and len(last_frame) == len(frame) else None
                if previous is None:
                    screen.bkgdset(' ', roles['text'])
                    screen.erase()
                for row, line in enumerate(frame):
                    if previous is not None and previous[row] == line:
                        continue
                    try:
                        if previous is not None:
                            screen.move(row, 0)
                            screen.clrtoeol()
                        screen.addstr(row, 0, line.text, roles[line.role])
                        if line.artwork_start is not None:
                            start = line.artwork_start
                            for column in range(start, len(line.text)):
                                glyph = line.text[column]
                                if glyph != ' ':
                                    shade = artwork_role(column - start, len(line.text) - start)
                                    screen.addstr(row, column, glyph, roles[shade])
                        for column, shade in line.highlights:
                            screen.addstr(row, column, line.text[column], roles[shade])
                    except curses.error:
                        # The emulator may resize between getmaxyx and addstr.
                        pass
                screen.noutrefresh()
                curses.doupdate()
                last_frame = frame
            try:
                key = screen.get_wch()
            except curses.error:
                continue
            items = matching_workflows(query)
            if key == curses.KEY_RESIZE:
                last_frame = None
                continue
            if key == '\x03':
                return 'q', palette, motion
            if not searching and isinstance(key, str):
                key = key.lower()
            if searching:
                if key in ('\n', '\r', curses.KEY_ENTER):
                    if items:
                        return items[selected % len(items)].key, palette, motion
                elif key == '\x1b':
                    searching, query = False, ''
                elif key in (curses.KEY_BACKSPACE, '\x7f', '\b'):
                    query = query[:-1]
                elif isinstance(key, str) and key.isprintable():
                    # ASCII search prevents ambiguous cell widths in the frame.
                    if key.isascii() and len(query) < 100:
                        query += key
                selected = 0
            elif key in ('q', '\x04'):
                return 'q', palette, motion
            elif key in ('\n', '\r', curses.KEY_ENTER):
                if items:
                    return items[selected % len(items)].key, palette, motion
            elif isinstance(key, str) and key in '12345678':
                return key, palette, motion
            elif key in (curses.KEY_DOWN, '\t', 'j'):
                selected = (selected + 1) % max(1, len(items))
            elif key in (curses.KEY_UP, curses.KEY_BTAB, 'k'):
                selected = (selected - 1) % max(1, len(items))
            elif key == '/':
                searching = True
            elif key == '\x1b':
                if help_open:
                    help_open = False
                else:
                    return 'q', palette, motion
            elif key == 't':
                palette = 'light' if palette == 'dark' else 'dark'
                roles = attributes()
                last_frame = None
            elif key == 'm':
                motion = not motion
                clock.set_running(motion)
                screen.timeout(round(1000 / ANIMATION_FPS) if motion else -1)
            elif key == '?':
                help_open = not help_open

    return curses.wrapper(session)
