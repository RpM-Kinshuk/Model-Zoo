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
    def without_stars(line):
        text = list(line.text)
        for column, _ in line.highlights:
            text[column] = " "
        return "".join(text)

    first = home.compose_frame(100, 32, 0, "", [], 0)
    second = home.compose_frame(100, 32, 0, "", [], 57)
    assert [without_stars(line) for line in first] == [without_stars(line) for line in second]


@pytest.mark.parametrize("keys", [b"\x1bOA\r", b"TM??/infrastructure\r"])
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
            'choice, _, motion = navigate(["Workspace not connected"], motion=False); '
            'print("RESULT", choice, "MOTION", motion)')
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
        assert (b'MOTION True' if b'M' in keys else b'MOTION False') in captured
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


def test_twinkle_fades_slowly_in_small_brightness_steps():
    ticks_per_second = home.ANIMATION_FPS
    for seed in (17, 120, 359):
        levels = [home.star_brightness(seed, tick) for tick in range(1 + 30 * ticks_per_second)]
        assert min(levels) == 0 and max(levels) == 63
        assert len(set(levels)) >= 24
        assert max(abs(a - b) for a, b in zip(levels, levels[1:])) <= 2
        period = next(tick for tick in range(18 * ticks_per_second, 30 * ticks_per_second + 1)
                      if all(home.star_brightness(seed, tick + offset) == levels[offset]
                             for offset in range(ticks_per_second)))
        assert 18 <= period / ticks_per_second <= 30


def test_braille_stars_stay_sparse_and_clear_of_workflows():
    frame = home.compose_frame(120, 48, 0, '', [], 500)
    dust = [line for line in frame if line.highlights]
    assert dust
    assert sum(len(line.highlights) for line in dust) < 120 * 48 / 50
    for line in dust:
        assert not any(item.title in line.text for item in home.WORKFLOWS)
        for column, _ in line.highlights:
            assert 0x2800 <= ord(line.text[column]) <= 0x28FF
            assert line.artwork_start is None or column < line.artwork_start


def test_home_keeps_galaxy_and_stars_at_standard_size():
    frame = home.compose_frame(80, 24, 0, "", ["Workspace  example"], 0)
    assert any("MODEL ZOO" in line.text for line in frame)
    assert any("✦" in line.text and line.artwork_start is not None for line in frame)
    assert any(line.highlights for line in frame[:8])
    assert any("Scan infrastructure" in line.text for line in frame)


def test_shared_header_fits_compact_terminal_sizes():
    for width, height in [(0, 2), (7, 10), (25, 12), (36, 16), (75, 24), (88, 40)]:
        header = home.brand_header(width, height)
        assert len(header) < height
        assert all(len(line.text) <= width for line in header)


def test_a_single_star_stays_in_place_until_it_has_gone_dark(monkeypatch):
    seed = 17
    monkeypatch.setattr(home, "_star_seeds", lambda field_seed, count: (seed,))
    field = [home.Line(" " * 36, "stars")] * 5
    visible_by_life = {}
    for tick in range(900):
        shown = home._scatter_stars(field, tick, 0)
        positions = [(row, column) for row, line in enumerate(shown) for column, _ in line.highlights]
        generation, _, _ = home._star_cycle(seed, tick)
        if home.star_brightness(seed, tick):
            assert len(positions) == 1
            visible_by_life.setdefault(generation, set()).update(positions)
        else:
            assert not positions
    assert all(len(positions) == 1 for positions in visible_by_life.values())
    assert len(set.union(*visible_by_life.values())) > 1


def test_motion_state_is_visible_in_the_home_controls():
    for active in (False, True):
        frame = home.compose_frame(80, 24, 0, "", [], 0, motion=active)
        assert any(("Motion on" if active else "Motion off") in line.text for line in frame)


def test_stars_have_a_dark_gap_between_fades():
    levels = [home.star_brightness(17, tick) for tick in range(home.ANIMATION_FPS * 20)]
    assert any(levels[index:index + home.ANIMATION_FPS] == [0] * home.ANIMATION_FPS
               for index in range(len(levels) - home.ANIMATION_FPS))


def test_starfields_change_location_and_density_with_time_and_seed():
    def positions(seed, tick):
        header = home.brand_header(75, 24, tick, star_seed=seed)
        return {(row, column) for row, line in enumerate(header) for column, _ in line.highlights}

    samples = [positions(17, tick) for tick in range(0, 600, 12)]
    assert len(set().union(*samples)) > max(map(len, samples))
    assert len(set(map(len, samples))) > 1
    assert positions(17, 24) == positions(17, 24)
    assert positions(17, 24) != positions(91, 24)


def test_stars_use_single_dots_and_keep_space_from_neighbors():
    from itertools import combinations

    for seed in (0, 17, 91):
        for tick in range(0, 480, 12):
            frame = home.compose_frame(80, 24, 0, "", [], tick, star_seed=seed)
            stars = []
            for row, line in enumerate(frame):
                for column, _ in line.highlights:
                    assert (ord(line.text[column]) - 0x2800).bit_count() == 1
                    stars.append((row, column))
            assert all(abs(a_row - b_row) > 1 or abs(a_col - b_col) > 3
                       for (a_row, a_col), (b_row, b_col) in combinations(stars, 2))


def test_brand_taglines_fit_the_home_frame():
    frame = home.compose_frame(80, 24, 0, "", [], 0)
    assert any(home.ATLAS_TAGLINE in line.text for line in frame[:-2])
    assert home.DISCOVERY_TAGLINE in frame[-2].text
    assert "Esc Quit" in frame[-2].text
