"""Shared progress-tracking helper for the long-running steps in this pipeline (DINOv3/DINOv2
feature precompute, training): a JSON status file + a human log line, written periodically and
atomically, read back by a dashboard. Same status shape as preprocessing.py's own progress.json
(phase/done/total/percent/rate/eta/started/updated/pid) so a dashboard can read either.
"""
import json
import logging
import os
import time
from datetime import datetime, timedelta


def format_duration(seconds):
    seconds = int(max(seconds, 0))
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, seconds = divmod(rest, 60)
    if days:
        return f"{days}d {hours}h {minutes:02d}m"
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m {seconds:02d}s"


class Progress:
    """progress = Progress(path, total=N, unit="clip"); progress.update(1); progress.write()

    `done` only counts work finished in this process's lifetime (not work resumed from a prior
    run), so the rate/ETA reflect actual current throughput rather than being skewed by however
    long a previous, now-finished run took.
    """

    def __init__(self, path, total, done=0, unit="item"):
        self.path = path
        self.total = total
        self.done = done
        self.start_done = done
        self.unit = unit
        self.started = time.time()
        self.started_at = datetime.now()
        self.last_write = 0.0

    def update(self, n=1):
        self.done += n

    def due(self, interval=10):
        return time.time() - self.last_write >= interval

    def write(self, phase="running", extra=None, log=True):
        self.last_write = time.time()
        now = datetime.now()
        elapsed = time.time() - self.started
        rate = (self.done - self.start_done) / elapsed if elapsed > 0 else 0.0
        remaining = self.total - self.done
        eta = remaining / rate if rate > 0 and remaining > 0 else None
        finish = now + timedelta(seconds=eta) if eta is not None else None
        status = {
            "phase": phase,
            "unit": self.unit,
            "done": self.done,
            "total": self.total,
            "percent": round(100 * self.done / self.total, 2) if self.total else 100.0,
            "rate": round(rate, 2),
            "eta": format_duration(eta) if eta is not None else None,
            "eta_finish": finish.isoformat(timespec="minutes") if finish else None,
            "started": self.started_at.isoformat(timespec="seconds"),
            "updated": now.isoformat(timespec="seconds"),
            "pid": os.getpid(),
            **(extra or {}),
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(status, indent=2))
        os.replace(tmp, self.path)

        if not log:
            return
        if eta is not None:
            logging.info(
                f"{phase}: {self.done:,}/{self.total:,} {self.unit}s ({status['percent']:.1f}%) | "
                f"{rate:.1f} {self.unit}s/s | ETA {status['eta']} (~{finish:%a %d.%m. %H:%M})"
            )
        else:
            logging.info(f"{phase}: {self.done:,}/{self.total:,} {self.unit}s ({status['percent']:.1f}%)")
