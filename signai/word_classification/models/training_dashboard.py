"""Live terminal dashboard for train.py: epoch/step progress, loss/accuracy, best checkpoint, ETA.

    python signai/word_classification/models/training_dashboard.py          # live view, Ctrl-C to close
    python signai/word_classification/models/training_dashboard.py --once   # print one snapshot

Only reads what train.py writes (checkpoints/v<N>/progress.json, state.json, the *.keras files), so it
can run in any terminal or a second tmux window, start and stop at will, and never affects training.
"""
import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from signai.word_classification.models.progress import format_duration  # noqa: E402
# from paths.py, not train.py: that would pull in all of TF/keras and the GPU-env re-exec, heavy
# for a dashboard that should start instantly
from signai.word_classification.paths import CHECKPOINT_DIR, MODEL_VERSION  # noqa: E402

PROGRESS_FILE = CHECKPOINT_DIR / "progress.json"
BEST_CKPT = CHECKPOINT_DIR / "word_classifier_best.keras"
LATEST_CKPT = CHECKPOINT_DIR / "word_classifier_latest.keras"
STALE_AFTER = 120
TITLE = f"[bold]SignAI[/] · word classifier v{MODEL_VERSION} training"
RUN_COMMAND = "python signai/word_classification/models/train.py"


def format_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def run_process(pid):
    try:
        proc = psutil.Process(pid)
        return proc if any("train.py" in part for part in proc.cmdline()) else None
    except psutil.Error:
        return None


def status_line(state, style, detail):
    return Text.assemble(("● ", style), (state, style), ("   " + detail, "dim"))


def stats_table():
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", no_wrap=True, min_width=14)
    table.add_column()
    return table


def fmt(x, spec=".4f"):
    return format(x, spec) if isinstance(x, (int, float)) else "--"


def system_row(table):
    mem = psutil.virtual_memory()
    load = psutil.getloadavg()[0]
    gpu = gpu_stats()
    line = f"CPU {psutil.cpu_percent():.0f}% · RAM {format_bytes(mem.used)} / {format_bytes(mem.total)} · load {load:.1f}"
    if gpu:
        line += f" · GPU {gpu}"
    table.add_row("System", line)


def gpu_stats():
    """nvidia-smi is simpler and more reliable here than a TF/torch GPU query (no process-level
    weirdness, no extra framework import for a dashboard that should stay lightweight)."""
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        )
        util, used, total = (x.strip() for x in out.stdout.strip().split(","))
        return f"{util}% util · {float(used)/1024:.1f}/{float(total)/1024:.1f} GB"
    except Exception:
        return None


def render():
    now = datetime.now()
    if not PROGRESS_FILE.exists():
        return Panel(
            Text(f"No training run found yet ({PROGRESS_FILE} doesn't exist).\n\nStart one with:\n  {RUN_COMMAND}"),
            title=TITLE, border_style="dim", width=96,
        )

    s = json.loads(PROGRESS_FILE.read_text())
    updated = datetime.fromisoformat(s["updated"])
    started = datetime.fromisoformat(s["started"])
    update_age = (now - updated).total_seconds()
    proc = run_process(s["pid"])

    if s["phase"] == "done":
        state, style, border = "DONE", "bold green", "green"
    elif proc is None:
        state, style, border = "STOPPED", "bold red", "red"
    elif update_age > STALE_AFTER:
        state, style, border = "NO UPDATES", "bold yellow", "yellow"
    else:
        state, style, border = "RUNNING", "bold green", "green"

    if proc:
        detail = f"pid {s['pid']}"
    elif s["phase"] == "done":
        detail = f"{s['epoch']}/{s['epochs_total']} epochs"
    else:
        detail = f"epoch {s['epoch']}/{s['epochs_total']} when it stopped"
    parts = [status_line(state, style, detail), Text()]

    epoch, epochs_total = s["epoch"], s["epochs_total"]
    bar = Table.grid(padding=(0, 2), expand=True)
    bar.add_column(ratio=1)
    bar.add_column(justify="right", no_wrap=True)
    bar.add_row(
        ProgressBar(total=epochs_total, completed=min(epoch, epochs_total), complete_style="cyan",
                    finished_style="green"),
        Text(f"epoch {epoch}/{epochs_total}", style="bold"),
    )
    parts += [bar, Text()]

    table = stats_table()
    if s["phase"] != "done" and s.get("steps_per_epoch"):
        step_bar = Table.grid(padding=(0, 1))
        step_bar.add_column()
        step_bar.add_column(no_wrap=True)
        step_bar.add_row(
            ProgressBar(total=s["steps_per_epoch"], completed=min(s["step"], s["steps_per_epoch"]),
                        width=40, complete_style="cyan"),
            f"{s['step']:,}/{s['steps_per_epoch']:,}",
        )
        table.add_row("Step", step_bar)
        if s.get("steps_per_sec"):
            table.add_row("Speed", f"{s['steps_per_sec']:.2f} steps/s")

    table.add_row("Loss / acc", f"{fmt(s.get('loss'))} / {fmt(s.get('accuracy'))}")
    table.add_row("Val loss/acc", f"{fmt(s.get('val_loss'))} / {fmt(s.get('val_accuracy'))}")
    if s.get("best_val_accuracy") is not None:
        table.add_row("Best val_acc", Text(f"{s['best_val_accuracy']:.4f} @ epoch {s['best_epoch']}", style="bold"))

    if s["phase"] == "done":
        table.add_row("Finished", f"{updated:%a %d.%m. %H:%M} · total {format_duration((updated - started).total_seconds())}")
    elif proc:
        if s.get("avg_epoch_seconds"):
            table.add_row("Avg epoch", format_duration(s["avg_epoch_seconds"]))
        if s.get("eta_epoch"):
            table.add_row("This epoch", f"ETA {s['eta_epoch']}")
        if s.get("eta_total"):
            table.add_row("Remaining", Text(f"{s['eta_total']}  (~{s['eta_total_finish']})", style="bold"))
    else:
        table.add_row("Stopped", f"{format_duration(update_age)} ago (last update {updated:%H:%M:%S})")

    for path, label in ((BEST_CKPT, "Best checkpoint"), (LATEST_CKPT, "Latest checkpoint")):
        if path.exists():
            age = time.time() - path.stat().st_mtime
            table.add_row(label, f"{format_bytes(path.stat().st_size)} · saved {format_duration(age)} ago")

    disk = shutil.disk_usage(CHECKPOINT_DIR)
    table.add_row("Disk", f"{format_bytes(disk.free)} free")
    system_row(table)
    parts.append(table)

    if proc is None and s["phase"] != "done":
        parts += [Text(), Text.assemble(("Resume with:  ", "dim"), (RUN_COMMAND, "bold"))]

    return Panel(
        Group(*parts), title=TITLE, border_style=border, width=96, padding=(1, 2),
        subtitle=f"[dim]{now:%H:%M:%S} · Ctrl-C closes this view, training keeps running[/]",
    )


def show(once=False):
    console = Console()
    psutil.cpu_percent()
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
