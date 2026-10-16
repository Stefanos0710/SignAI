"""Live terminal dashboard for preprocessing.py: progress, speed, ETA, checkpoints, disk, system.

    python signai/word_classification/preprocessing_dashboard.py          # live view, Ctrl-C to close
    python signai/word_classification/preprocessing_dashboard.py --once   # print one snapshot

Only reads what preprocessing.py writes (progress.json and shards/), so it can run in any terminal
or a second tmux window, start and stop at will, and never affects the run itself.
"""
import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from signai.word_classification.paths import PROCESSED_DIR  # noqa: E402

# must match preprocessing.py's PROGRESS_FILE / SHARD_DIR (not imported: that would load MediaPipe)
PROGRESS_FILE = PROCESSED_DIR / "progress.json"
SHARD_DIR = PROCESSED_DIR / "shards"
STALE_AFTER = 120  # seconds without a progress.json update before a live run looks stuck
SPARK_CHARS = "▁▂▃▄▅▆▇█"
SPARK_WIDTH = 48
TITLE = "[bold]SignAI[/] · word-clip preprocessing"
RUN_COMMAND = "python signai/word_classification/preprocessing.py"

_shard_counts = {}  # (name, mtime) -> clips in that shard, so each shard is opened only once


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


def format_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def sparkline(values, width=SPARK_WIDTH):
    values = values[-width:]
    top = max(values, default=0)
    if top <= 0:
        return ""
    return "".join(SPARK_CHARS[round(v / top * (len(SPARK_CHARS) - 1))] for v in values)


def shard_history():
    """-> [(mtime, clips, bytes)] per checkpoint shard, oldest first."""
    history = []
    for path in SHARD_DIR.glob("shard_*.npz"):
        stat = path.stat()
        key = (path.name, stat.st_mtime)
        if key not in _shard_counts:
            with np.load(path) as f:
                _shard_counts[key] = len(f["names"]) + len(f["failed_names"])
        history.append((stat.st_mtime, _shard_counts[key], stat.st_size))
    return sorted(history)


def checkpoint_rates(history, started):
    """clips/s between consecutive checkpoints of the current run (the first one is measured from
    the run's start, so it includes worker startup)."""
    run = [(t, n) for t, n, _ in history if t >= started]
    times = [started] + [t for t, _ in run]
    return [n / (t1 - t0) for (t1, n), t0 in zip(run, times) if t1 > t0]


def run_process(pid):
    """The preprocessing process if it's still alive (checks the command line too, so a reused
    pid isn't mistaken for the run), else None."""
    try:
        proc = psutil.Process(pid)
        return proc if any("preprocessing.py" in part for part in proc.cmdline()) else None
    except psutil.Error:
        return None


def status_line(state, style, detail):
    return Text.assemble(("● ", style), (state, style), ("   " + detail, "dim"))


def stats_table():
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", no_wrap=True, min_width=11)
    table.add_column()
    return table


def system_row(table):
    mem = psutil.virtual_memory()
    load = psutil.getloadavg()[0]
    table.add_row(
        "System",
        f"CPU {psutil.cpu_percent():.0f}% · RAM {format_bytes(mem.used)} / {format_bytes(mem.total)} "
        f"· load {load:.1f}",
    )


def render():
    now = datetime.now()
    if not PROGRESS_FILE.exists():
        return Panel(
            Text(f"No preprocessing run found yet ({PROGRESS_FILE} doesn't exist).\n\nStart one with:\n  {RUN_COMMAND}"),
            title=TITLE, border_style="dim", width=96,
        )

    s = json.loads(PROGRESS_FILE.read_text())
    started = datetime.fromisoformat(s["started"])
    updated = datetime.fromisoformat(s["updated"])
    update_age = (now - updated).total_seconds()
    proc = run_process(s["pid"])
    workers = len(proc.children()) if proc else 0

    if s["phase"] == "done":
        state, style, border = "DONE", "bold green", "green"
    elif proc is None:
        state, style, border = "STOPPED", "bold red", "red"
    elif s["phase"] != "extracting":
        state, style, border = "MERGING", "bold yellow", "yellow"
    elif update_age > STALE_AFTER:
        state, style, border = "NO UPDATES", "bold yellow", "yellow"
    else:
        state, style, border = "RUNNING", "bold green", "green"

    if s["phase"] == "done":
        detail = "train/val/test splits written"
    else:
        detail = s["phase"] + (f" · pid {s['pid']} · {workers} workers" if proc else "")
    parts = [status_line(state, style, detail), Text()]

    done, total, failed = s["done"], s["total"], s["failed"]
    bar = Table.grid(padding=(0, 2), expand=True)
    bar.add_column(ratio=1)
    bar.add_column(justify="right", no_wrap=True)
    bar.add_row(
        ProgressBar(total=total, completed=done, complete_style="cyan", finished_style="green"),
        Text(f"{100 * done / total:5.1f}%", style="bold"),
    )
    parts += [bar, Text()]

    table = stats_table()
    failed_text = Text(f"{failed:,} failed ({100 * failed / max(done, 1):.2f}%)",
                       style="red" if failed else "dim")
    table.add_row("Clips", Text.assemble(f"{done:,} / {total:,}   ", failed_text))

    history = shard_history()
    rates = checkpoint_rates(history, started.timestamp())

    if s["phase"] == "extracting" and proc is None:
        table.add_row("Ran", f"{started:%a %H:%M} – {updated:%a %H:%M} "
                             f"({format_duration((updated - started).total_seconds())}), "
                             f"stopped {format_duration(update_age)} ago")
        if s["clips_per_sec"]:
            table.add_row("Speed", f"{s['clips_per_sec']:.1f} clips/s avg")
    elif s["phase"] == "extracting":
        recent = rates[-3:]
        current = sum(recent) / len(recent) if recent else s["clips_per_sec"]
        table.add_row("Speed", f"{current:.1f} clips/s now · {s['clips_per_sec']:.1f} avg this run")
        table.add_row("Elapsed", f"{format_duration((now - started).total_seconds())} "
                                 f"(started {started:%a %H:%M})")
        if s["eta_finish"]:
            finish = datetime.fromisoformat(s["eta_finish"])
            table.add_row("Remaining", Text(format_duration((finish - now).total_seconds()), style="bold"))
            table.add_row("Finishes", Text.assemble((f"{finish:%a %d.%m. %H:%M}", "bold"),
                                                    ("   then merge + split", "dim")))
        else:
            table.add_row("Remaining", Text("measuring…", style="dim"))
    elif s["phase"] == "done":
        table.add_row("Finished", f"{updated:%a %d.%m. %H:%M} · took {format_duration((updated - started).total_seconds())}")
    else:
        table.add_row("Merging", f"shards -> train/val/test, since {updated:%H:%M} ({format_duration(update_age)})")

    if s["phase"] == "extracting" and rates:
        table.add_row("History", Text.assemble((sparkline(rates), "cyan"),
                                               (f"  clips/s per checkpoint, last {min(len(rates), SPARK_WIDTH)}", "dim")))

    disk = shutil.disk_usage(PROCESSED_DIR)
    if history:
        shard_bytes = sum(b for _, _, b in history)
        last_age = time.time() - history[-1][0]
        table.add_row("Checkpoints", f"{len(history)} shards · {format_bytes(shard_bytes)} · "
                                     f"last saved {format_duration(last_age)} ago")
    if history and s["phase"] == "extracting":
        clips_saved = sum(n for _, n, _ in history)
        still_needed = shard_bytes / clips_saved * (total - clips_saved)
        disk_style = "red" if still_needed > disk.free else "default"
        table.add_row("Disk", Text(f"{format_bytes(disk.free)} free · shards need ~{format_bytes(still_needed)} more",
                                   style=disk_style))
    else:
        table.add_row("Disk", f"{format_bytes(disk.free)} free")
    system_row(table)
    parts.append(table)

    if s["phase"] == "done":
        outputs = sorted(PROCESSED_DIR.glob("*_data.npz")) + sorted(PROCESSED_DIR.glob("*_images.npz"))
        files = stats_table()
        for path in outputs:
            files.add_row(path.name, format_bytes(path.stat().st_size))
        parts += [Text(), Text("Output", style="bold"), files]
    elif proc is None:
        parts += [Text(), Text.assemble(("Resume from the last checkpoint with:  ", "dim"), (RUN_COMMAND, "bold"))]
    elif state == "NO UPDATES":
        parts += [Text(), Text(f"No progress update for {format_duration(update_age)} -- check the tmux session",
                               style="yellow")]

    return Panel(
        Group(*parts), title=TITLE, border_style=border, width=96, padding=(1, 2),
        subtitle=f"[dim]{now:%H:%M:%S} · Ctrl-C closes this view, preprocessing keeps running[/]",
    )


def show(once=False):
    console = Console()
    psutil.cpu_percent()  # first call only primes the counter
    if once:
        time.sleep(0.3)
        console.print(render())
        return
    try:
        with Live(render(), console=console, refresh_per_second=2, screen=False) as live:
            while True:
                time.sleep(1)
                live.update(render())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--once", action="store_true", help="print one snapshot and exit")
    show(once=parser.parse_args().once)
