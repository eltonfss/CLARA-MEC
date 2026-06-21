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
import yaml
from datetime import datetime

CONFIG_FILE = "config.yaml"

# Ler dataset diretamente do YAML
with open(CONFIG_FILE, "r") as f:
    config = yaml.safe_load(f)

dataset = config["dataset"]["name"]
baseline = config["experiment"]["name"]


class MetricsLogger:

    def __init__(self, results_dir="results"):

        os.makedirs(results_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.csv_path = os.path.join(
            results_dir,
            f"{baseline}_metrics_{timestamp}_{dataset}.csv"
        )

        self.metrics = []

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

        return result

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