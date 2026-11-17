"""Responsive, single-owner terminal navigator; workflow execution lives in tui.

The frame builder is pure. Only the event loop touches the terminal, and curses
restores its modes before any subprocess or ordinary line prompt runs.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
import unicodedata
from typing import Sequence


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
    width, height = max(0, width - 1), max(0, height)
    if not height:
        return []
    items = matching_workflows(query)
    selected = min(max(0, selected), max(0, len(items) - 1))
    if height < 5:
        title = items[selected].title if items else 'No matches'
        rows = [Line('MODEL ZOO', 'brand'), Line(title, 'selected'), Line('Resize for more · q quit', 'muted')]
        return [Line(_fit_cells(row.text, width), row.role) for row in rows[:height]]
    rows = [Line('  M O D E L   Z O O', 'brand')]
    if height >= 34 and width >= 72:
        rows = [
            Line('                             ·          ✦                ·   ✧', 'stars'),
            Line('  ✦  M O D E L   Z O O                  .  ·   .', 'brand'),
            Line('                                  ·´    ✧    .    `·', 'accent'),
            Line('  A living atlas of evidence    ∘    .     ✦     .    ∘', 'brand'),
            Line('                                  `·.    .    ✧  .´', 'accent'),
            Line('  Collect. Measure. Discover.          ` ·  · ´', 'muted'),
            Line(''),
        ]
    elif height >= 16:
        rows += [Line('  A living atlas of model evidence', 'accent')]
    if height >= 22:
        # Stable positions: only brightness/glyph changes, never surrounding text.
        stars = [' '] * width
        for position in range(5, width, 11):
            # Different periods and phase offsets give each star its own breath.
            period = 37 + position % 23
            phase = (tick + position * 7) % period
            envelope = min(phase, period - phase)
            stars[position] = '✦' if envelope < 2 else '✧' if envelope < 5 else '·' if envelope < 11 else '.'
        # A rare short streak passes through the decorative band only.
        comet = tick % 173
        if 0 < comet < min(width - 3, 22):
            for offset, glyph in enumerate('··✦'):
                stars[comet + offset] = glyph
        rows += [Line(''.join(stars), 'stars'), Line('')]
    if height >= 16:
        rows += [Line('  ' + line, 'muted') for line in status[:2]]
        rows += [Line('')]
    rows += [Line('  SEARCH / ' + query + ('_' if searching else '')
                  if query or searching else '  YOUR NEXT DISCOVERY', 'accent')]
    footer = [Line('  Up/Down select · Enter open · / search · q quit', 'muted'),
              Line(f'  {selected + 1 if items else 0}/{len(items)} workflows · t theme · m motion · ? help', 'muted')]
    if searching:
        footer = [Line('  Type to filter · Enter open · Esc clear', 'accent'),
                  Line('  Search titles and workflow descriptions', 'muted')]
    elif help_open:
        footer = [Line('  1-8 open directly · Esc clears search', 'accent'),
                  Line('  Tab next · t palette · m animation · ? close help', 'muted')]
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
            rows += [Line(f' {marker} {item.key}  {item.title}', 'selected' if index == selected else 'text')]
            if detailed:
                rows += [Line('      ' + item.description, 'muted')]
            if spacious:
                rows.append(Line(''))
    while len(rows) < height - len(footer):
        rows.append(Line(''))
    rows += footer
    margin = ' ' * min(8, max(0, (width - 78) // 2))
    return [Line(_fit_cells(margin + row.text, width), row.role) for row in rows[:height]]


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
        screen.timeout(100)
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
                     'stars': curses.A_NORMAL, 'glow': curses.A_BOLD, 'spark': curses.A_BOLD, 'selected': curses.A_REVERSE | curses.A_BOLD}
            if colors_available:
                # Explicit background makes contrast independent of emulator theme.
                light = palette == 'light'
                background = curses.COLOR_WHITE if light else curses.COLOR_BLACK
                foreground = curses.COLOR_BLACK if light else curses.COLOR_WHITE
                accent = curses.COLOR_BLUE if light else curses.COLOR_CYAN
                secondary, muted, selection = curses.COLOR_MAGENTA, foreground, accent
                glow, spark = curses.COLOR_CYAN, curses.COLOR_YELLOW
                selected_foreground = background
                if curses.COLORS >= 256:
                    # Luminous jewel accents on deep space; saturated ink on cool white.
                    background, foreground, accent, secondary, muted, selection = (
                        (255, 234, 27, 91, 240, 189) if light else
                        (233, 255, 87, 177, 250, 54))
                    glow, spark = (26, 130) if light else (123, 221)
                    selected_foreground = foreground
                for index, (fg, bg) in enumerate(((foreground, background), (accent, background),
                                                 (secondary, background), (selected_foreground, selection),
                                                 (muted, background), (glow, background),
                                                 (spark, background)), 1):
                    curses.init_pair(index, fg, bg)
                roles.update(text=curses.color_pair(1), muted=curses.color_pair(5),
                             brand=curses.color_pair(2) | curses.A_BOLD,
                             accent=curses.color_pair(3) | curses.A_BOLD,
                             glow=curses.color_pair(6) | curses.A_BOLD,
                             spark=curses.color_pair(7) | curses.A_BOLD,
                             stars=curses.color_pair(3), selected=curses.color_pair(4) | curses.A_BOLD)
                screen.bkgd(' ', roles['text'])
            return roles

        roles = attributes()
        while True:
            height, width = screen.getmaxyx()
            tick = int(time.monotonic() / 0.12) if motion else 0
            frame = compose_frame(width, height, selected, query, status, tick, help_open=help_open, searching=searching)
            if frame != last_frame:
                screen.erase()
                for row, line in enumerate(frame):
                    try:
                        screen.addstr(row, 0, line.text, roles[line.role])
                        if line.role == 'stars':
                            for column, glyph in enumerate(line.text):
                                if glyph in '✧✦':
                                    screen.addstr(row, column, glyph,
                                                  roles['spark' if glyph == '✦' else 'glow'])
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
