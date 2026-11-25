"""Responsive, single-owner terminal navigator; workflow execution lives in tui.

The frame builder is pure. Only the event loop touches the terminal, and curses
restores its modes before any subprocess or ordinary line prompt runs.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
import math
import random
from functools import lru_cache
import unicodedata
from typing import Sequence

from zoo.terminal_style import PALETTES, PageLayout


@dataclass(frozen=True)
class Workflow:
    key: str
    title: str
    description: str
    group: str


WORKFLOWS = (
    Workflow('1', 'Prepare a collection', 'Pin model revisions and review ready / held rows', 'COLLECT'),
    Workflow('2', 'Plan a measurement run', 'Choose resources, inspect preflight, review command', 'COLLECT'),
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


def star_brightness(position: int, tick: int) -> int:
    """Sixty-four luminance steps over slow, independently phased cycles."""
    phase = tick / (110 + position % 89) + position * 1.73
    wave = (1 + math.sin(phase)) / 2
    return round(63 * wave * wave * (3 - 2 * wave))


@lru_cache(maxsize=384)
def star_color(light: bool, family: int, level: int) -> int:
    """Quantize a smooth fade to the terminal's fixed 256-color palette.

    Keep the emulator's color definitions untouched. Repeated indices are normal
    where terminal color depth cannot express adjacent animation steps.
    """
    background = (238, 238, 238) if light else (18, 18, 18)
    targets = ((24, 75, 145), (145, 55, 20), (100, 45, 145)) if light else (
        (220, 239, 255), (255, 190, 120), (185, 155, 255))
    amount = level / 63
    target = tuple(a + (b-a) * amount for a, b in zip(background, targets[family]))
    steps = (0, 95, 135, 175, 215, 255)
    candidates = [(16 + 36*r + 6*g + b, (red, green, blue))
                  for r, red in enumerate(steps) for g, green in enumerate(steps)
                  for b, blue in enumerate(steps)]
    candidates += [(232 + index, (8 + index*10,) * 3) for index in range(24)]
    return min(candidates, key=lambda item: sum((a-b)**2 for a, b in zip(item[1], target)))[0]


@lru_cache(maxsize=256)
def dust_locations(width: int, row: int) -> tuple[tuple[int, int], ...]:
    """Independent row samples avoid the diagonal bands of modular placement.

    Local seeded randomness keeps layout and fade phases stable across frames;
    sampling without replacement also prevents two stars sharing a cell.
    """
    if width < 8:
        return ()
    generator = random.Random(20261028 + row * 65537)
    columns = generator.sample(range(3, width - 3), 2)
    return tuple((column, generator.randrange(1_000_000)) for column in columns)


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
                  status: Sequence[str], tick: int, *, help_open: bool = False, searching: bool = False) -> list[Line]:
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
    rows = [Line('MODEL ZOO', 'brand')]
    if height >= 34 and width >= 72:
        art = spiral_galaxy()
        artwork_start = width - len(art[0])
        rows = [Line(('MODEL ZOO' if index == 1 else '').ljust(artwork_start) + glyphs,
                     'brand' if index == 1 else 'text', artwork_start=artwork_start)
                for index, glyphs in enumerate(art)]
    elif height >= 16:
        rows.insert(0, Line(''))
    if height >= 18 and status:
        rows += [Line(line, 'muted') for line in status[:2]]
        rows.append(Line(''))
    rows += [Line('Search  ' + query + ('_' if searching else '')
                  if query or searching else 'Workflows', 'accent')]
    if width >= 48:
        footer = [Line('↑/↓ Select  ·  Enter Open  ·  / Search  ·  Q Quit', 'muted'),
                  Line(f'{selected + 1 if items else 0}/{len(items)}  ·  T Theme  ·  M Motion  ·  ? Help', 'muted')]
    elif width >= 30:
        footer = [Line('↑/↓ Select  ·  Enter Open', 'muted'), Line('/ Search  ·  Q Quit', 'muted')]
    else:
        footer = [Line('↑↓ Enter / Q', 'muted')]
    if searching:
        footer = [Line('Type to filter  ·  Enter Open', 'accent'), Line('Esc Clear search', 'muted')]
    elif help_open:
        footer = [Line('1–8 Open directly  ·  Esc Clear search', 'accent'),
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
    while len(rows) < height - len(footer):
        row_index = len(rows)
        if width >= 50 and height >= 22:
            dust = [' '] * width
            highlights = []
            # A fixed spatial seed prevents flicker; slow envelopes reveal stars
            # at apparently random times. Only unused padding receives dust.
            for slot, (column, seed) in enumerate(dust_locations(width, row_index)):
                brightness = star_brightness(seed, tick)
                if brightness > 25:
                    level = round((brightness - 26) * 63 / 37)
                    dust[column] = chr(0x2800 + (1, 8, 16, 64, 128, 9)[seed % 6])
                    highlights.append((column, f'dust{slot}_{level}'))
            rows.append(Line(''.join(dust), 'stars', highlights=tuple(highlights)))
        else:
            rows.append(Line(''))
    rows += footer
    margin = layout.indent
    return [Line(margin + _fit_cells(row.text, width), row.role,
                 row.artwork_start + len(margin) if row.artwork_start is not None else None,
                 tuple((column + len(margin), role) for column, role in row.highlights
                       if column < width))
            for row in rows[:height]]


def navigate(status: Sequence[str], *, palette: str = 'dark', motion: bool = True,
             color: bool = True) -> tuple[str, str, bool]:
    """Run an alternate-screen session and return a workflow and preferences."""
    import curses

    def session(screen):
        nonlocal palette, motion
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        screen.keypad(True)
        screen.timeout(16)
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
                    roles[name] = curses.A_NORMAL
                    if colors_available:
                        if curses.COLORS >= 256 and curses.COLOR_PAIRS >= 202:
                            pair = 10 + family * 64 + level
                            shade = star_color(palette == 'light', family, level)
                            curses.init_pair(pair, shade, background)
                            roles[name] = curses.color_pair(pair)
                        else:
                            roles[name] = roles['muted' if level < 32 else 'spark']
            return roles

        roles = attributes()
        while True:
            height, width = screen.getmaxyx()
            tick = int(time.monotonic() * 60) if motion else 0
            frame = compose_frame(width, height, selected, query, status, tick, help_open=help_open, searching=searching)
            if frame != last_frame:
                screen.bkgdset(' ', roles['text'])
                screen.erase()
                for row, line in enumerate(frame):
                    try:
                        screen.addstr(row, 0, line.text, roles[line.role])
                        if line.artwork_start is not None:
                            spectrum = ('glow', 'accent', 'rose', 'spark', 'aurora')
                            for column in range(line.artwork_start, len(line.text)):
                                glyph = line.text[column]
                                if glyph != ' ':
                                    # Cool outer arms transition to a warm stellar core.
                                    distance = abs(column - line.artwork_start - 16)
                                    shade = 'spark' if distance < 3 else spectrum[min(4, distance // 4)]
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
                query, selected = '', 0
            elif key == 't':
                palette = 'light' if palette == 'dark' else 'dark'
                roles = attributes()
                last_frame = None
            elif key == 'm':
                motion = not motion
            elif key == '?':
                help_open = not help_open

    return curses.wrapper(session)
