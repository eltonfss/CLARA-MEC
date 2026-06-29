"""Run CLARA-MEC convergence hypothesis sweeps.

The script materializes a set of config.yaml variants, runs train.py for each
variant, copies the generated metrics to a stable experiment directory, and
creates summary CSV/PNG visualizations for convergence comparison.
"""

from __future__ import annotations

import argparse
import copy
import csv
import glob
import os
import shutil
import subprocess
import sys
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


def set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    current = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def read_losses(path: Path) -> list[float]:
    with path.open(newline="") as f:
        return [float(row["loss"]) for row in csv.DictReader(f) if row.get("loss") not in (None, "")]


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=None, help="Override rounds for quicker smoke sweeps.")
    parser.add_argument("--num-samples", type=int, default=None, help="Override dataset.num_samples for quicker smoke sweeps.")
    parser.add_argument("--dry-run", action="store_true", help="Only write variant configs and hypothesis manifest; do not train.")
    args = parser.parse_args()

    with CONFIG_PATH.open() as f:
        base_config = yaml.safe_load(f)

    run_dir = SWEEP_DIR / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    configs_dir = run_dir / "configs"
    metrics_dir = run_dir / "metrics"
    configs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    metrics_files: dict[str, Path] = {}
    original_config = CONFIG_PATH.read_text()

    try:
        for hypothesis in HYPOTHESES:
            cfg = copy.deepcopy(base_config)
            for key, value in hypothesis["overrides"].items():
                set_nested(cfg, key, value)
            if args.rounds is not None:
                set_nested(cfg, "experiment.rounds", args.rounds)
            if args.num_samples is not None:
                set_nested(cfg, "dataset.num_samples", args.num_samples)

            config_file = configs_dir / f"{hypothesis['name']}.yaml"
            with config_file.open("w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)

            manifest.append({"name": hypothesis["name"], "description": hypothesis["description"], "config": str(config_file.relative_to(ROOT))})
            if args.dry_run:
                continue

            CONFIG_PATH.write_text(config_file.read_text())
            before = set(glob.glob(str(RESULTS_DIR / "*_metrics_*.csv")))
            subprocess.run([sys.executable, "train.py"], cwd=ROOT, check=True)
            generated = latest_metrics(before)
            if generated is None:
                raise RuntimeError(f"No metrics CSV generated for {hypothesis['name']}")
            stable_metrics = metrics_dir / f"{hypothesis['name']}.csv"
            shutil.copy2(generated, stable_metrics)
            metrics_files[hypothesis["name"]] = stable_metrics
    finally:
        CONFIG_PATH.write_text(original_config)

    with (run_dir / "hypotheses.yaml").open("w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    if not args.dry_run:
        summary_file = run_dir / "summary.csv"
        with summary_file.open("w", newline="") as f:
            fieldnames = ["hypothesis", "initial_loss", "final_loss", "best_loss", "loss_delta", "loss_std", "oscillation_rate", "converged", "metrics_file"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for name, metrics_file in metrics_files.items():
                row = {"hypothesis": name, **convergence_stats(read_losses(metrics_file)), "metrics_file": str(metrics_file.relative_to(ROOT))}
                writer.writerow(row)
        plot_losses(metrics_files, run_dir / "loss_curves.png")
        print(f"Sweep complete: {run_dir.relative_to(ROOT)}")
    else:
        print(f"Dry run complete: {run_dir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
