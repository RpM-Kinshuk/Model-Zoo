"""Behavioral checks for the responsive terminal navigator."""
import pytest

from zoo import terminal_home as home


def test_every_frame_fits_even_tiny_windows():
    for width in (1, 8, 24, 40, 80, 120):
        for height in (1, 4, 8, 12, 24, 40):
            frame = home.compose_frame(width, height, 7, '', ['Workspace not connected'], 0)
            assert len(frame) <= height
            assert all(len(line.text) < width for line in frame)


def test_filter_finds_workflow_by_intent():
    assert [item.key for item in home.matching_workflows('fingerprint')] == ['7']
    assert home.matching_workflows('no such workflow') == []


def test_selected_workflow_stays_visible_in_small_window():
    frame = home.compose_frame(50, 8, 7, '', [], 0)
    assert any('Scan infrastructure' in line.text for line in frame)


def test_motion_does_not_change_content_rows():
    first = home.compose_frame(100, 32, 0, '', [], 0)
    second = home.compose_frame(100, 32, 0, '', [], 3)
    assert [line.text for line in first if line.role != 'stars'] == [line.text for line in second if line.role != 'stars']


@pytest.mark.parametrize('keys', [b'\x1bOA\r', b'TM??/infrastructure\r'])
def test_real_terminal_resize_navigation_and_mode_restoration(keys):
    """Exercise curses itself, not a mocked screen, through a pseudo-terminal."""
    import os
    import sys
    import time
    import subprocess
    import select
    import signal
    import struct
    import pytest

    if os.name != 'posix':
        pytest.skip('POSIX pseudo-terminal test')
    import fcntl
    import pty
    import termios

    master, slave = pty.openpty()
    original = termios.tcgetattr(slave)
    environment = dict(os.environ, TERM='xterm-256color')
    code = ('from zoo.terminal_home import navigate; '
            'print("RESULT", navigate(["Workspace not connected"], motion=False)[0])')
    process = subprocess.Popen([sys.executable, '-c', code], stdin=slave,
                               stdout=slave, stderr=slave, env=environment)
    captured = bytearray()

    def drain(duration):
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            if select.select([master], [], [], 0.02)[0]:
                captured.extend(os.read(master, 65536))

    try:
        drain(0.3)
        for rows, columns in [(8, 24), (1, 1), (40, 120), (12, 40), (24, 80)]:
            fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', rows, columns, 0, 0))
            process.send_signal(signal.SIGWINCH)
            drain(0.15)
        # Last workflow is the infrastructure scan; End is not needed to reach it.
        os.write(master, keys)
        drain(0.3)
        process.wait(timeout=5)
        drain(0.1)
        assert process.returncode == 0, captured.decode(errors='replace')
        assert b'RESULT 8' in captured
        assert termios.tcgetattr(slave) == original
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        os.close(master)
        os.close(slave)


def test_unicode_workspace_paths_fit_terminal_cells():
    from zoo.tui import _display_width
    frame = home.compose_frame(40, 24, 0, '', ['Data root: /' + '界' * 30], 0)
    assert all(_display_width(line.text) < 40 for line in frame)


def test_plain_menu_exposes_every_choice_in_short_terminal(monkeypatch, capsys):
    import os
    from zoo import tui
    monkeypatch.setattr(tui.shutil, 'get_terminal_size', lambda fallback: os.terminal_size((40, 10)))
    tui._draw_home(None)
    output = capsys.readouterr().out
    assert '8  Scan infrastructure' in output
    assert 'Up/Down' not in output
    assert 'Enter a number' in output


def test_home_workflows_follow_number_order():
    assert [item.key for item in home.matching_workflows('')] == list('12345678')


def test_spacious_frame_separates_workflows():
    frame = home.compose_frame(120, 48, 0, '', [], 0)
    first = next(index for index, line in enumerate(frame) if 'Prepare a collection' in line.text)
    assert not frame[first + 2].text.strip()


def test_galaxy_color_region_never_overlaps_header_copy():
    frame = home.compose_frame(110, 40, 0, '', [], 0)
    line = next(line for line in frame if 'MODEL ZOO' in line.text)
    assert line.artwork_start is not None
    assert line.text.index('MODEL ZOO') + len('MODEL ZOO') < line.artwork_start


def test_spiral_galaxy_has_distinct_arms_and_bright_core():
    artwork = home.spiral_galaxy()
    assert len(artwork) == 9
    assert '✦' in artwork[4]
    assert all(len(line) == 33 for line in artwork)
    assert sum(line.count(' ') for line in artwork) > 150


def test_menu_uses_multiple_warm_and_cool_accents():
    frame = home.compose_frame(110, 40, 0, '', [], 0)
    roles = {line.role for line in frame if any(item.title in line.text for item in home.WORKFLOWS)}
    assert {'spark', 'aurora', 'rose'}.issubset(roles)


def test_twinkle_has_gradual_brightness_steps():
    levels = [home.star_brightness(17, tick) for tick in range(600)]
    assert len(set(levels)) >= 10
    assert max(abs(a-b) for a, b in zip(levels, levels[1:])) <= 1


def test_twinkle_has_many_steps_without_speeding_up():
    levels = [home.star_brightness(17, tick) for tick in range(1800)]
    assert len(set(levels)) == 64
    assert max(abs(a-b) for a, b in zip(levels, levels[1:])) <= 1


def test_braille_stars_stay_sparse_and_clear_of_workflows():
    frame = home.compose_frame(120, 48, 0, '', [], 500)
    dust = [line for line in frame if any(role.startswith('dust') for _, role in line.highlights)]
    assert dust
    assert all(sum(not char.isspace() for char in line.text) <= 2 for line in dust)
    assert all(line.role == 'stars' for line in dust)


def test_dust_scatter_has_no_repeating_diagonal_stride():
    from collections import Counter
    for width in (50, 80, 120):
        positions = [home.dust_locations(width, row) for row in range(40)]
        assert positions == [home.dust_locations(width, row) for row in range(40)]
        assert all(len({column for column, _ in row}) == 2 for row in positions)
        for slot in range(2):
            strides = Counter((b[slot][0] - a[slot][0]) % (width - 6)
                              for a, b in zip(positions, positions[1:]))
            assert max(strides.values()) < 8
