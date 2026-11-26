"""Terminal rendering regressions for the interactive launcher."""

import os
import sys
from types import ModuleType

import pytest

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


def test_workflow_header_keeps_terminal_scrollback(capsys):
    tui._workflow_header("1")
    output = capsys.readouterr().out
    assert "\x1b[?1049" not in output
    assert "COLLECT" in output


def test_all_workflow_titles_align_with_home_brand(monkeypatch, capsys):
    from zoo.terminal_home import WORKFLOWS, compose_frame

    monkeypatch.setattr(tui.shutil, 'get_terminal_size', lambda fallback: os.terminal_size((106, 24)))
    frame = compose_frame(106, 24, 0, '', [], 0)
    home_brand = next(line.text for line in frame if 'MODEL ZOO' in line.text)
    gutter = len(home_brand) - len(home_brand.lstrip())
    for workflow in WORKFLOWS:
        tui._workflow_header(workflow.key)
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


def test_clear_screen_resets_viewport_and_preserves_scrollback(monkeypatch, capsys):
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(tui.os, "name", "posix")
    monkeypatch.setattr(tui.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(tui.sys.stdout, "isatty", lambda: True)
    tui._clear_screen()
    assert capsys.readouterr().out == "\x1b[2J\x1b[H"


def test_clear_screen_leaves_redirected_output_plain(monkeypatch, capsys):
    monkeypatch.setattr(tui.sys.stdout, "isatty", lambda: False)
    tui._clear_screen()
    assert capsys.readouterr().out == ""


def test_escape_cancels_an_interactive_workflow_before_launch(tmp_path, monkeypatch, capsys):
    catalog = tmp_path / "builds" / "current"
    catalog.mkdir(parents=True)
    (catalog / "models.parquet").touch()
    answers = iter(("6", "\x1b", "q"))
    monkeypatch.setattr(tui, "_read_input", lambda *args, **kwargs: next(answers))
    monkeypatch.setattr(tui.subprocess, "run", lambda *args, **kwargs: pytest.fail("unexpected launch"))
    assert tui.main(["--plain", "--workspace", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "Start this workflow?" in output
    assert output.count("MODEL ZOO") == 2


def test_windows_console_editor_supports_arrows_and_escape(monkeypatch):
    console = ModuleType("msvcrt")
    keys = iter(("a", "c", "\xe0", "K", "b", "\ud83d", "\ude00", "\r", "\x1b"))
    console.getwch = lambda: next(keys)
    monkeypatch.setitem(sys.modules, "msvcrt", console)
    monkeypatch.setattr(tui.os, "name", "nt")
    assert tui._raw_terminal_input("Input ", None) == "ab😀c"
    with pytest.raises(tui.Back):
        tui._raw_terminal_input("Input ", None)


@pytest.mark.skipif(os.name != "posix", reason="POSIX pseudo-terminal test")
@pytest.mark.parametrize("plain", [True, False])
def test_escape_navigates_search_back_to_home_in_a_real_terminal(tmp_path, plain):
    import fcntl
    import pty
    import select
    import struct
    import subprocess
    import termios
    import time

    catalog = tmp_path / "builds" / "current"
    catalog.mkdir(parents=True)
    (catalog / "models.parquet").touch()
    master, slave = pty.openpty()
    original_mode = termios.tcgetattr(slave)
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 32, 100, 0, 0))
    command = [sys.executable, "-m", "zoo.tui", "--workspace", str(tmp_path)]
    if plain:
        command.append("--plain")
    process = subprocess.Popen(command, stdin=slave, stdout=slave, stderr=slave,
                               env=dict(os.environ, TERM="xterm-256color"))
    output = bytearray()

    def expect(marker):
        deadline = time.monotonic() + 5
        while marker not in output and time.monotonic() < deadline:
            if select.select([master], [], [], 0.05)[0]:
                output.extend(os.read(master, 65536))
        assert marker in output, output.decode(errors="replace")
        del output[:output.index(marker) + len(marker)]

    def send(key):
        os.write(master, key)

    try:
        expect(b"Workflow [" if plain else b"Esc Quit")
        send(b"7\r" if plain else b"7")
        expect(b"Search by name")
        send(b"1\r")
        expect(b"Collect. Measure. Discover.")
        expect(b"Search text")
        expect(b"\x1b7\x1b[")  # Header cells update while the input waits.
        send(b"\x1b[D")  # Left arrow must not trigger Back.
        send(b"/nested" * 20 + b"unsubmitted")
        expect(b"unsubmitted")
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 8, 32, 0, 0))
        expect(b"\x1b[2;7r")
        expect(b"Esc Back")
        expect(b"unsubmitted")
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 32, 100, 0, 0))
        expect(b"Collect. Measure. Discover.")
        expect(b"unsubmitted")
        send(b"\x1b")
        expect(b"Search by name")
        send(b"\x1b")
        expect(b"Workflow [" if plain else b"Esc Quit")
        send(b"\x1b")
        expect(b"See you next time.")
        assert process.wait(timeout=5) == 0
        assert termios.tcgetattr(slave) == original_mode
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        os.close(slave)
        os.close(master)


def test_clear_screen_uses_native_command_in_legacy_windows_console(monkeypatch):
    from unittest.mock import Mock

    command = Mock()
    monkeypatch.setattr(tui.os, "name", "nt")
    monkeypatch.setattr(tui.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(tui.sys.stdout, "isatty", lambda: True)
    monkeypatch.delenv("WT_SESSION", raising=False)
    monkeypatch.delenv("TERM", raising=False)
    monkeypatch.setattr(tui.subprocess, "run", command)
    tui._clear_screen()
    command.assert_called_once_with(["cmd.exe", "/d", "/c", "cls"], check=False)


def test_workflow_page_reuses_galaxy_header_with_vertical_spacing(monkeypatch, capsys):
    monkeypatch.setattr(tui.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(tui.sys.stdout, "isatty", lambda: True)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(tui.shutil, "get_terminal_size", lambda fallback: os.terminal_size((80, 24)))
    monkeypatch.setattr(tui.THEME, "enabled", False)
    tui._workflow_header("7")
    output = capsys.readouterr().out
    assert "MODEL ZOO" in output
    assert "✦" in output
    assert "\n\n" in output
    assert output.index("MODEL ZOO") < output.index("Explore the zoo")


def test_numbered_menu_keeps_brand_visible_in_short_windows(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(tui.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(tui.sys.stdout, "isatty", lambda: True)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(tui.shutil, "get_terminal_size", lambda fallback: os.terminal_size((80, 16)))
    monkeypatch.setattr(tui.THEME, "enabled", False)
    tui._draw_home(tmp_path)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) < 16  # Leave the last row for the prompt.
    assert any("MODEL ZOO" in line for line in lines)
    assert any("Scan infrastructure" in line for line in lines)


@pytest.mark.parametrize(("flags", "enabled"), [([], True), (["--no-motion"], False), (["--motion"], True)])
def test_motion_setting_works_without_color_and_survives_navigation(tmp_path, monkeypatch, flags, enabled):
    from zoo import terminal_home

    calls = []

    def navigate(status, *, palette, motion, color, star_seed, clock):
        calls.append((motion, color, star_seed))
        return ("8" if len(calls) == 1 else "q", palette, motion)

    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("TERM", "xterm-256color")
    if tui.os.name == "nt":
        monkeypatch.setenv("WT_SESSION", "test")
    monkeypatch.setattr(tui.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(tui.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(terminal_home, "navigate", navigate)
    monkeypatch.setattr(tui, "_infrastructure", lambda workspace: 0)
    monkeypatch.setattr(tui, "_pause", lambda: None)
    assert tui.main([*flags, "--workspace", str(tmp_path)]) == 0
    assert calls == [(enabled, False, calls[0][2]), (enabled, False, calls[0][2])]
