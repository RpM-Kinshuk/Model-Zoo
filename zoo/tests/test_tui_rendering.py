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
    monkeypatch.setattr(tui.shutil, 'get_terminal_size', lambda fallback: os.terminal_size((80, 24)))
    layout = tui._layout()
    assert layout.width + 2 * layout.gutter == 79


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
    theme = tui.Theme(palette='dark')
    theme.enabled = True
    assert theme.paint('text', 'brand') == '\x1b[1;38;5;81mtext\x1b[0m'
    monkeypatch.setenv('COLORFGBG', '0;15')
    assert theme.automatic_palette() == 'light'
    theme.palette = 'light'
    assert theme.paint('text', 'brand') == '\x1b[1;38;5;25mtext\x1b[0m'


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


def test_all_workflow_titles_align_with_home_brand(monkeypatch, capsys):
    from zoo.terminal_home import WORKFLOWS, compose_frame

    monkeypatch.setattr(tui.shutil, 'get_terminal_size', lambda fallback: os.terminal_size((106, 24)))
    frame = compose_frame(106, 24, 0, '', [], 0)
    home_brand = next(line.text for line in frame if 'MODEL ZOO' in line.text)
    gutter = len(home_brand) - len(home_brand.lstrip())
    for workflow in WORKFLOWS:
        with tui._workflow_screen(workflow.key, enabled=False):
            pass
        lines = capsys.readouterr().out.splitlines()
        assert ' ' * gutter + workflow.title in lines
        assert all(tui._display_width(line) < 106 for line in lines)


def test_workflow_help_and_errors_wrap_inside_the_content_column(monkeypatch, capsys):
    monkeypatch.setattr(tui.shutil, 'get_terminal_size', lambda fallback: os.terminal_size((36, 20)))
    tui._write("A long prompt with 中文 text should remain readable in a compact terminal.", "muted")
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) > 1
    assert all(line.startswith('  ') and tui._display_width(line) < 36 for line in lines)
    assert ' '.join(line.strip() for line in lines).endswith('compact terminal.')


def test_long_default_wraps_above_a_short_prompt(monkeypatch, capsys):
    monkeypatch.setattr(tui.shutil, 'get_terminal_size', lambda fallback: os.terminal_size((36, 20)))
    prompts = []
    monkeypatch.setattr(tui, '_read_input', lambda prompt, completion: prompts.append(prompt) or '')
    default = '/workspace/' + 'nested/' * 10 + 'models.csv'
    assert tui._ask('Input CSV', default) == default
    output = capsys.readouterr().out.splitlines()
    assert all(tui._display_width(line) < 36 for line in output)
    assert ''.join(line[2:] for line in output[2:]) == 'Default  ' + default
    assert prompts == ['  ›: ']


def test_literal_wrapping_preserves_spaces_inside_shell_arguments():
    command = "python 'path with   repeated spaces.csv' --prepare_only"
    assert ''.join(tui.wrap_cells(command, 16, literal=True)) == command


def test_redirected_input_keeps_prompt_and_status_on_separate_lines(monkeypatch, capsys):
    from io import StringIO

    monkeypatch.setattr(tui.sys, 'stdin', StringIO('1\n'))
    assert tui._read_menu_choice(None) == '1'
    tui._write('Prepare a collection')
    assert len(capsys.readouterr().out.splitlines()) == 2
