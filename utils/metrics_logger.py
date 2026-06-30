"""
===============================================================================
metrics_logger.py

Gerenciamento e exportação das métricas do CLARA-MEC.

Registra métricas por round federado e exporta CSV para análise e geração
de gráficos do artigo.

Métricas registradas:
- loss
- accuracy
- latency
- energy
- cost

Autor: Nelson Machado Junior
===============================================================================
"""

import os
import csv
import time
import yaml
from datetime import datetime

CONFIG_FILE = "config.yaml"

# Ler dataset diretamente do YAML
with open(CONFIG_FILE, "r") as f:
    config = yaml.safe_load(f)

dataset = config["dataset"]["name"]
baseline = config["experiment"]["name"]


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


class MetricsLogger:

    def __init__(self, results_dir="results"):

        os.makedirs(results_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.csv_path = os.path.join(
            results_dir,
            f"{baseline}_metrics_{timestamp}_{dataset}.csv"
        )

        self.metrics = []
        self.started_at = time.monotonic()
        self.total_rounds = int(config.get("experiment", {}).get("rounds", 0) or 0)

        self.fieldnames = [
            "round",
            "loss",
            "accuracy",
            "latency",
            "energy",
            "cost"
        ]

    # -------------------------------------------------------------------------
    # Agregação das métricas recebidas do Flower
    # -------------------------------------------------------------------------

    # def aggregate_fit_metrics(self, metrics):

    #     aggregated = {}

    #     for num_examples, client_metrics in metrics:

    #         for key, value in client_metrics.items():
    #             aggregated.setdefault(key, []).append(value)

    #     result = {
    #         k: sum(v) / len(v)
    #         for k, v in aggregated.items()
    #     }

    #     self.fit_metrics.append(result)

    #     return result
    
    # def aggregate_metrics(self, metrics):

    #     aggregated = {}

    #     for num_examples, client_metrics in metrics:

    #         for key, value in client_metrics.items():
    #             aggregated.setdefault(key, []).append(value)

    #     result = {
    #         k: sum(v) / len(v)
    #         for k, v in aggregated.items()
    #     }

    #     self.metrics.append(result)

    #     return result

    def aggregate_metrics(self, metrics):

        aggregated = {}
        total_examples = 0

        for num_examples, client_metrics in metrics:

            total_examples += num_examples

            for key, value in client_metrics.items():
                aggregated.setdefault(key, 0.0)
                aggregated[key] += value * num_examples

        result = {
            k: v / total_examples
            for k, v in aggregated.items()
        }

        self.metrics.append(result)
        self.save()
        self.print_round_progress(result)

        return result

    def print_round_progress(self, result):
        current_round = len(self.metrics)
        total_rounds = self.total_rounds or current_round
        percent = min(100.0, current_round / total_rounds * 100)
        elapsed = time.monotonic() - self.started_at
        eta = None
        if current_round and self.total_rounds and current_round < self.total_rounds:
            eta = elapsed / current_round * (self.total_rounds - current_round)

        print(
            "📈 Round "
            f"{current_round}/{total_rounds} ({percent:.1f}%) "
            f"loss={result.get('loss', 0):.6f} "
            f"accuracy={result.get('accuracy', 0):.4f} "
            f"elapsed={format_duration(elapsed)} "
            f"eta={format_duration(eta) if eta is not None else 'done'}",
            flush=True,
        )

    # -------------------------------------------------------------------------
    # Salvar métricas em arquivo CSV
    # -------------------------------------------------------------------------

    def save(self):

        if not self.metrics:
            print("Nenhuma métrica registrada.")
            return

        with open(self.csv_path, "w", newline="") as f:

            writer = csv.DictWriter(
                f,
                fieldnames=self.fieldnames
            )

            writer.writeheader()

            for i, row in enumerate(self.metrics):

                row_data = {
                    "round": i + 1,
                    "loss": row.get("loss", 0),
                    "accuracy": row.get("accuracy", 0),
                    "latency": row.get("latency", 0),
                    "energy": row.get("energy", 0),
                    "cost": row.get("cost", 0),
                }

                writer.writerow(row_data)

        print(f"📊 Métricas salvas em: {self.csv_path}")
