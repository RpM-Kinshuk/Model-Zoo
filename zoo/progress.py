"""Count real work using terminal-safe progress lines."""

import time


class Progress:
    def __init__(self, label, total, emit):
        self.label = label
        self.total = total
        self.emit = emit
        self.last = None
        self.last_time = 0.0

    def update(self, finished, *, running=None, final=False):
        if self.total == 0:
            return
        width = 20
        filled = width * finished // self.total
        mark = (filled, running)
        now = time.monotonic()
        if mark == self.last and not final and now - self.last_time < 60:
            return
        if final and self.last == mark and finished == self.total:
            return
        bar = "#" * filled + "-" * (width - filled)
        line = f"{self.label} [{bar}] {finished}/{self.total} finished"
        if running is not None:
            pending = "not started" if final else "waiting"
            line += f" | {running} running | {self.total - finished - running} {pending}"
        try:
            self.emit(line)
        except OSError:
            return  # A disconnected output stream should not interrupt the job.
        self.last, self.last_time = mark, now
