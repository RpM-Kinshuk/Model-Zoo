"""Terminal rendering regressions for the interactive launcher."""

import os
import sys
from types import ModuleType

from zoo import tui


def test_windows_full_screen_is_available_without_term_when_curses_exists(monkeypatch):
    monkeypatch.setattr(tui.os, 'name', 'nt')
    monkeypatch.setattr(tui.sys.stdin, 'isatty', lambda: True)
    monkeypatch.setattr(tui.sys.stdout, 'isatty', lambda: True)
    monkeypatch.delenv('TERM', raising=False)
    monkeypatch.setitem(sys.modules, 'curses', ModuleType('curses'))
    assert tui._supports_full_screen()


def test_full_screen_stays_disabled_for_dumb_terminal(monkeypatch):
    monkeypatch.setattr(tui.sys.stdin, 'isatty', lambda: True)
    monkeypatch.setattr(tui.sys.stdout, 'isatty', lambda: True)
    monkeypatch.setenv('TERM', 'dumb')
    assert not tui._supports_full_screen()


def test_frame_reserves_last_terminal_column(monkeypatch):
    monkeypatch.setattr(tui, '_FRAME_SIZE', os.terminal_size((80, 24)))
    assert tui._width() == 79


def test_query_limits_are_validated(monkeypatch):
    answers = iter(['zero', '0', '5'])
    monkeypatch.setattr(tui, '_ask', lambda *args, **kwargs: next(answers))
    assert tui._ask_integer('Maximum results', '10', minimum=1) == '5'


def test_home_fits_common_terminal_sizes(monkeypatch, capsys):
    for columns, rows in [(40, 20), (80, 24), (100, 28), (100, 34), (120, 44)]:
        monkeypatch.setattr(tui.shutil, 'get_terminal_size', lambda fallback: os.terminal_size((columns, rows)))
        tui._draw_home(None)
        lines = capsys.readouterr().out.splitlines()
        assert len(lines) < rows, (columns, rows, len(lines))
        assert all(tui._display_width(line) < columns for line in lines)


def test_theme_enables_color_and_detects_light_background(monkeypatch):
    theme = tui.Theme()
    theme.enabled = True
    assert theme.paint('text', '38;5;73') == '\x1b[38;5;81mtext\x1b[0m'
    monkeypatch.setenv('COLORFGBG', '0;15')
    assert theme.automatic_palette() == 'light'
    theme.palette = 'light'
    assert theme.paint('text', '38;5;73') == '\x1b[38;5;25mtext\x1b[0m'


def test_workflow_screen_restores_terminal_after_error(monkeypatch, capsys):
    import pytest
    monkeypatch.setenv('TERM', 'xterm-256color')
    monkeypatch.setattr(tui.sys.stdout, 'isatty', lambda: True)
    with pytest.raises(RuntimeError):
        with tui._workflow_screen('1'):
            raise RuntimeError('cancelled')
    output = capsys.readouterr().out
    assert '\x1b[?1049h\x1b[2J\x1b[H' in output
    assert output.endswith('\x1b[?1049l')
    assert 'COLLECT' in output
