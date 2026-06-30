"""Run resumable CLARA-MEC convergence hypothesis sweeps.

The script materializes config.yaml variants, runs train.py for each variant,
copies generated metrics to a stable experiment directory, and creates summary
CSV/PNG visualizations for convergence comparison. It writes a state.json
checkpoint after every transition so interrupted sweeps can continue with
``--resume latest`` or ``--resume <run_dir>``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - handled at runtime
    plt = None

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"
RESULTS_DIR = ROOT / "results"
SWEEP_DIR = ROOT / "results" / "aggregation_hypothesis_sweep"
STATE_FILE = "state.json"

HYPOTHESES: list[dict[str, Any]] = [
    {
        "name": "baseline_noniid_sparse_clients",
        "description": "Current setup: non-IID split, 50 clients, 10% participation, and 30 local epochs. High drift and only five selected clients per round can make loss oscillate.",
        "overrides": {},
    },
    {
        "name": "higher_participation_noniid",
        "description": "Increase participation to 50% while keeping non-IID data. More client updates per round should reduce aggregation variance.",
        "overrides": {"federated_learning.client_fraction": 0.5},
    },
    {
        "name": "all_clients_noniid",
        "description": "Use all clients each round under non-IID data. This tests whether convergence issues are mainly due to sparse client sampling.",
        "overrides": {"federated_learning.client_fraction": 1.0},
    },
    {
        "name": "fewer_clients_higher_ratio_noniid",
        "description": "Use fewer total clients and a higher ratio so each selected update has more local data and the server sees a broader population per round.",
        "overrides": {"federated_learning.total_clients": 20, "federated_learning.client_fraction": 0.5},
    },
    {
        "name": "iid_sparse_clients",
        "description": "Keep sparse participation but switch to IID sampling. If this converges better, label/client heterogeneity is a major cause.",
        "overrides": {"dataset.split": "iid"},
    },
    {
        "name": "noniid_less_local_drift",
        "description": "Keep non-IID data but reduce local epochs from 30 to 5. This tests whether excessive local training causes client drift before aggregation.",
        "overrides": {"federated_learning.local_epochs": 5},
    },
    {
        "name": "fedprox_noniid_stabilized",
        "description": "Switch to FedProx with fewer local epochs and higher participation. Proximal regularization should reduce non-IID client drift.",
        "overrides": {
            "experiment.strategy": "fedprox",
            "experiment.name": "FedProx-Stability",
            "federated_learning.client_fraction": 0.5,
            "federated_learning.local_epochs": 5,
        },
    },
]

FAST_HYPOTHESES = [
    "baseline_noniid_sparse_clients",
    "higher_participation_noniid",
    "iid_sparse_clients",
]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    current = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def read_losses(path: Path) -> list[float]:
    with path.open(newline="") as f:
        return [float(row["loss"]) for row in csv.DictReader(f) if row.get("loss") not in (None, "")]


def count_metric_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def convergence_stats(losses: list[float]) -> dict[str, float | bool]:
    if not losses:
        return {"initial_loss": 0.0, "final_loss": 0.0, "best_loss": 0.0, "loss_delta": 0.0, "loss_std": 0.0, "oscillation_rate": 0.0, "converged": False}
    increases = sum(1 for prev, cur in zip(losses, losses[1:]) if cur > prev)
    mean = sum(losses) / len(losses)
    variance = sum((x - mean) ** 2 for x in losses) / len(losses)
    final_window = losses[-max(3, len(losses) // 5):]
    initial_window = losses[: max(3, len(losses) // 5)]
    converged = (sum(final_window) / len(final_window)) < (sum(initial_window) / len(initial_window)) and increases / max(1, len(losses) - 1) <= 0.5
    return {
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "best_loss": min(losses),
        "loss_delta": losses[0] - losses[-1],
        "loss_std": variance ** 0.5,
        "oscillation_rate": increases / max(1, len(losses) - 1),
        "converged": converged,
    }


def latest_metrics(before: set[str]) -> Path | None:
    after = set(glob.glob(str(RESULTS_DIR / "*_metrics_*.csv")))
    new_files = [Path(p) for p in after - before]
    if not new_files:
        return None
    return max(new_files, key=lambda p: p.stat().st_mtime)


def plot_losses(metrics_files: dict[str, Path], output: Path) -> None:
    if plt is None:
        print("matplotlib is unavailable; skipping plot generation", file=sys.stderr)
        return
    plt.figure(figsize=(12, 7))
    for name, metrics_file in metrics_files.items():
        losses = read_losses(metrics_file)
        plt.plot(range(1, len(losses) + 1), losses, marker="o", linewidth=1.5, label=name)
    plt.xlabel("Federated round")
    plt.ylabel("Weighted evaluation loss")
    plt.title("CLARA-MEC aggregation convergence hypotheses")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def resolve_resume_dir(value: str) -> Path:
    if value == "latest":
        candidates = [p for p in SWEEP_DIR.glob("*") if (p / STATE_FILE).exists()]
        if not candidates:
            raise FileNotFoundError("No resumable sweep found under results/aggregation_hypothesis_sweep")
        return max(candidates, key=lambda p: p.stat().st_mtime)
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def save_state(run_dir: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = now_iso()
    write_json(run_dir / STATE_FILE, state)


def load_state(run_dir: Path) -> dict[str, Any]:
    with (run_dir / STATE_FILE).open() as f:
        return json.load(f)


def progress_line(state: dict[str, Any]) -> str:
    total = len(state["selected_hypotheses"])
    completed = len(state["completed"])
    failed = len(state["failed"])
    remaining = max(0, total - completed)
    pending = max(0, total - completed - failed)
    percent = (completed / total * 100) if total else 100.0
    durations = [item["duration_seconds"] for item in state["completed"].values() if item.get("duration_seconds")]
    eta = None
    if durations and remaining:
        eta = (sum(durations) / len(durations)) * remaining
    return (
        f"[sweep] progress: {completed}/{total} hypotheses complete "
        f"({percent:.1f}%), {pending} pending, {failed} failed/retryable, "
        f"{remaining} total remaining, ETA {format_duration(eta)}"
    )


def create_config(base_config: dict[str, Any], hypothesis: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = copy.deepcopy(base_config)
    for key, value in hypothesis["overrides"].items():
        set_nested(cfg, key, value)
    if args.rounds is not None:
        set_nested(cfg, "experiment.rounds", args.rounds)
    if args.num_samples is not None:
        set_nested(cfg, "dataset.num_samples", args.num_samples)
    if args.local_epochs is not None:
        set_nested(cfg, "federated_learning.local_epochs", args.local_epochs)
    return cfg


def initialize_run(args: argparse.Namespace, selected_hypotheses: list[dict[str, Any]], base_config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    run_dir = SWEEP_DIR / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    configs_dir = run_dir / "configs"
    metrics_dir = run_dir / "metrics"
    configs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for hypothesis in selected_hypotheses:
        cfg = create_config(base_config, hypothesis, args)
        config_file = configs_dir / f"{hypothesis['name']}.yaml"
        with config_file.open("w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        manifest.append({"name": hypothesis["name"], "description": hypothesis["description"], "config": str(config_file.relative_to(ROOT))})

    with (run_dir / "hypotheses.yaml").open("w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    state = {
        "run_dir": str(run_dir.relative_to(ROOT)),
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "status": "initialized",
        "selected_hypotheses": [h["name"] for h in selected_hypotheses],
        "completed": {},
        "failed": {},
        "current": None,
        "dry_run": args.dry_run,
    }
    save_state(run_dir, state)
    return run_dir, state


def collect_completed_metrics(run_dir: Path, state: dict[str, Any]) -> dict[str, Path]:
    metrics_files: dict[str, Path] = {}
    for name, payload in state.get("completed", {}).items():
        metrics_path = ROOT / payload["metrics_file"]
        if metrics_path.exists():
            metrics_files[name] = metrics_path
    return metrics_files


def run_hypothesis(run_dir: Path, state: dict[str, Any], name: str, config_file: Path, expected_rounds: int) -> Path:
    print("\n" + "=" * 88, flush=True)
    print(f"[sweep] starting hypothesis: {name}", flush=True)
    print(progress_line(state), flush=True)
    print(f"[sweep] config: {config_file.relative_to(ROOT)}", flush=True)
    print(f"[sweep] expected rounds in this hypothesis: {expected_rounds}", flush=True)

    before = set(glob.glob(str(RESULTS_DIR / "*_metrics_*.csv")))
    CONFIG_PATH.write_text(config_file.read_text())
    hypothesis_started = time.monotonic()
    state["current"] = {"name": name, "started_at": now_iso(), "config": str(config_file.relative_to(ROOT)), "expected_rounds": expected_rounds}
    state["status"] = "running"
    save_state(run_dir, state)

    process = subprocess.Popen([sys.executable, "train.py"], cwd=ROOT)
    try:
        return_code = process.wait()
    except KeyboardInterrupt:
        print("\n[sweep] KeyboardInterrupt received. Terminating active training process safely...", flush=True)
        process.terminate()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            print("[sweep] training process did not stop after 60s; killing it.", flush=True)
            process.kill()
            process.wait()
        partial = latest_metrics(before)
        if partial is not None:
            partial_dir = run_dir / "partial_metrics"
            partial_dir.mkdir(exist_ok=True)
            partial_copy = partial_dir / f"{name}.partial.csv"
            shutil.copy2(partial, partial_copy)
            state["current"]["partial_metrics_file"] = str(partial_copy.relative_to(ROOT))
        state["status"] = "interrupted"
        save_state(run_dir, state)
        raise

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, [sys.executable, "train.py"])

    generated = latest_metrics(before)
    if generated is None:
        raise RuntimeError(f"No metrics CSV generated for {name}")

    stable_metrics = run_dir / "metrics" / f"{name}.csv"
    shutil.copy2(generated, stable_metrics)
    rows = count_metric_rows(stable_metrics)
    if rows < expected_rounds:
        print(f"[sweep] warning: {name} produced {rows}/{expected_rounds} metric rows.", flush=True)

    duration = time.monotonic() - hypothesis_started
    state["completed"][name] = {
        "completed_at": now_iso(),
        "duration_seconds": duration,
        "metrics_file": str(stable_metrics.relative_to(ROOT)),
        "rounds_completed": rows,
        "rounds_expected": expected_rounds,
    }
    state["current"] = None
    state["status"] = "running"
    save_state(run_dir, state)
    print(f"[sweep] completed {name} in {format_duration(duration)} with {rows}/{expected_rounds} rounds.", flush=True)
    print(progress_line(state), flush=True)
    return stable_metrics


def finalize_outputs(run_dir: Path, state: dict[str, Any], metrics_files: dict[str, Path]) -> None:
    if not metrics_files:
        print("[sweep] no completed metrics available; summary and plot skipped.", flush=True)
        return

    summary_file = run_dir / "summary.csv"
    with summary_file.open("w", newline="") as f:
        fieldnames = ["hypothesis", "initial_loss", "final_loss", "best_loss", "loss_delta", "loss_std", "oscillation_rate", "converged", "metrics_file"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, metrics_file in metrics_files.items():
            row = {"hypothesis": name, **convergence_stats(read_losses(metrics_file)), "metrics_file": str(metrics_file.relative_to(ROOT))}
            writer.writerow(row)
    plot_losses(metrics_files, run_dir / "loss_curves.png")
    state["summary_file"] = str(summary_file.relative_to(ROOT))
    state["plot_file"] = str((run_dir / "loss_curves.png").relative_to(ROOT))
    save_state(run_dir, state)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=None, help="Override rounds for quicker smoke sweeps.")
    parser.add_argument("--num-samples", type=int, default=None, help="Override dataset.num_samples for quicker smoke sweeps.")
    parser.add_argument("--local-epochs", type=int, default=None, help="Override federated_learning.local_epochs for every generated config.")
    parser.add_argument("--hypotheses", default="all", help="Comma-separated hypothesis names to run, or 'all'.")
    parser.add_argument("--fast", action="store_true", help="Run a small, quick diagnostic sweep: 3 rounds, 1000 samples, 3 local epochs, and three representative hypotheses.")
    parser.add_argument("--dry-run", action="store_true", help="Only write variant configs and hypothesis manifest; do not train.")
    parser.add_argument("--resume", default=None, help="Resume a previous run directory, or use 'latest'.")
    args = parser.parse_args()

    with CONFIG_PATH.open() as f:
        base_config = yaml.safe_load(f)

    hypothesis_names = {h["name"] for h in HYPOTHESES}
    if args.resume:
        run_dir = resolve_resume_dir(args.resume)
        state = load_state(run_dir)
        selected_names = set(state["selected_hypotheses"])
        print(f"[sweep] resuming run: {run_dir.relative_to(ROOT)}", flush=True)
    else:
        if args.fast:
            args.rounds = args.rounds or 3
            args.num_samples = args.num_samples or 1000
            args.local_epochs = args.local_epochs or 3
            if args.hypotheses == "all":
                args.hypotheses = ",".join(FAST_HYPOTHESES)

        selected_names = hypothesis_names
        if args.hypotheses != "all":
            selected_names = {name.strip() for name in args.hypotheses.split(",") if name.strip()}
            unknown_names = selected_names - hypothesis_names
            if unknown_names:
                raise ValueError(f"Unknown hypotheses: {', '.join(sorted(unknown_names))}")
        selected_hypotheses = [h for h in HYPOTHESES if h["name"] in selected_names]
        run_dir, state = initialize_run(args, selected_hypotheses, base_config)
        print(f"[sweep] initialized run: {run_dir.relative_to(ROOT)}", flush=True)

    selected_hypotheses = [h for h in HYPOTHESES if h["name"] in selected_names]
    metrics_files = collect_completed_metrics(run_dir, state)
    original_config = CONFIG_PATH.read_text()

    if state.get("dry_run"):
        state["status"] = "dry_run_complete"
        save_state(run_dir, state)
        print(f"[sweep] dry run complete: {run_dir.relative_to(ROOT)}", flush=True)
        return 0

    try:
        for hypothesis in selected_hypotheses:
            name = hypothesis["name"]
            if name in state.get("completed", {}):
                print(f"[sweep] skipping completed hypothesis: {name}", flush=True)
                continue

            config_file = run_dir / "configs" / f"{name}.yaml"
            if not config_file.exists():
                cfg = create_config(base_config, hypothesis, args)
                with config_file.open("w") as f:
                    yaml.safe_dump(cfg, f, sort_keys=False)

            with config_file.open() as f:
                cfg = yaml.safe_load(f)
            expected_rounds = int(cfg["experiment"]["rounds"])

            try:
                state.get("failed", {}).pop(name, None)
                metrics_files[name] = run_hypothesis(run_dir, state, name, config_file, expected_rounds)
                finalize_outputs(run_dir, state, metrics_files)
            except KeyboardInterrupt:
                print(f"[sweep] interrupted. Resume with: python experiments/aggregation_hypothesis_sweep.py --resume {run_dir.relative_to(ROOT)}", flush=True)
                return 130
            except Exception as exc:  # keep the sweep resumable after unexpected failures
                state["failed"][name] = {"failed_at": now_iso(), "error": repr(exc)}
                state["current"] = None
                state["status"] = "failed"
                save_state(run_dir, state)
                print(f"[sweep] hypothesis failed: {name}: {exc!r}", flush=True)
                print(f"[sweep] state checkpoint saved. Resume after fixing the issue with: python experiments/aggregation_hypothesis_sweep.py --resume {run_dir.relative_to(ROOT)}", flush=True)
                raise
    finally:
        CONFIG_PATH.write_text(original_config)

    state["status"] = "complete"
    save_state(run_dir, state)
    finalize_outputs(run_dir, state, metrics_files)
    print(f"[sweep] complete: {run_dir.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
