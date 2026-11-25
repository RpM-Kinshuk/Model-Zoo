from zoo.progress import Progress
from zoo.terminal_style import PALETTES, Theme


def test_progress_logs_actual_counts_and_interrupted_tail():
    lines = []
    progress = Progress("Measurement", 5, lines.append)
    progress.update(0, running=2)
    progress.update(2, running=1)
    progress.update(2, running=0, final=True)
    assert lines[0] == "Measurement [--------------------] 0/5 finished | 2 running | 3 waiting"
    assert lines[1] == "Measurement [########------------] 2/5 finished | 1 running | 2 waiting"
    assert lines[-1] == "Measurement [########------------] 2/5 finished | 0 running | 3 not started"
    assert all("\r" not in line and "\x1b" not in line for line in lines)


def test_progress_reports_terminal_changes_but_limits_large_batch_logs():
    lines = []
    progress = Progress("Review", 1000, lines.append)
    for finished in range(1001):
        progress.update(finished)
    assert len(lines) == 21 and "1000/1000 finished" in lines[-1]


def test_explicit_theme_uses_the_shared_palette_and_honors_no_color(monkeypatch):
    theme = Theme(palette="light", enabled=True)
    assert theme.paint("Title", "brand") == f"\x1b[1;38;5;{PALETTES['light']['brand']}mTitle\x1b[0m"
    monkeypatch.setenv("NO_COLOR", "")
    assert Theme(palette="light").paint("Title", "brand") == "Title"
