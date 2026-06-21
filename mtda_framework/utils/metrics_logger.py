"""
===============================================================================
metrics_logger.py

Logger científico de métricas do MTDA.
===============================================================================
"""

import csv
import os
from datetime import datetime


class MetricsLogger:

    def __init__(self, save_dir="results"):

        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )

        self.csv_path = os.path.join(
            save_dir,
            f"mtda_metrics_{timestamp}.csv"
        )

        self.metrics = []

        self.fieldnames = [
            "epoch",
            "train_loss",
            "test_loss",
            "accuracy_offloading",
            "accuracy_resource",
            "latency",
            "energy",
            "cost",
            "forgetting_score",
            "domain_gap"
        ]

    # ==========================================================
    # Registrar métricas
    # ==========================================================
    def log(
        self,
        epoch,
        train_loss,
        test_loss,
        accuracy_offloading,
        accuracy_resource=0.0,
        latency=0.0,
        energy=0.0,
        cost=0.0,
        forgetting_score=0.0,
        domain_gap=0.0
    ):

        self.metrics.append({

            "epoch": int(epoch),

            "train_loss": float(train_loss),

            "test_loss": float(test_loss),

            "accuracy_offloading": float(
                accuracy_offloading
            ),

            "accuracy_resource": float(
                accuracy_resource
            ),

            "latency": float(latency),

            "energy": float(energy),

            "cost": float(cost),

            "forgetting_score": float(
                forgetting_score
            ),

            "domain_gap": float(
                domain_gap
            )
        })

    # # ==========================================================
    # # Salvar CSV
    # # ==========================================================
    # def save(self):

    #     if len(self.metrics) == 0:

    #         print(
    #             "\n⚠ Nenhuma métrica registrada."
    #         )
    #         return

    #     with open(
    #         self.csv_path,
    #         "w",
    #         newline=""
    #     ) as csvfile:

    #         writer = csv.DictWriter(
    #             csvfile,
    #             fieldnames=self.fieldnames
    #         )

    #         writer.writeheader()

    #         for row in self.metrics:
    #             writer.writerow(row)

    #     print(
    #         f"\n📊 Métricas salvas em:\n"
    #         f"{self.csv_path}"
    #     )


    # ==========================================================
    # Salvar CSV
    # ==========================================================
    def save(self, history=None):

        # ==========================================
        # Caso venha history do treinamento
        # ==========================================
        if history is not None:

            self.metrics = []

            for epoch in range(len(history["loss"])):

                self.log(
                    epoch=epoch + 1,

                    train_loss=history["loss"][epoch],

                    test_loss=history["loss"][epoch],

                    accuracy_offloading=history[
                        "accuracy"
                    ][epoch],

                    accuracy_resource=0.0,

                    latency=0.0,

                    energy=0.0,

                    cost=0.0,

                    forgetting_score=0.0,

                    domain_gap=0.0
                )

        # ==========================================
        # Salvar CSV
        # ==========================================
        if len(self.metrics) == 0:

            print(
                "\n⚠ Nenhuma métrica registrada."
            )
            return

        with open(
            self.csv_path,
            "w",
            newline=""
        ) as csvfile:

            writer = csv.DictWriter(
                csvfile,
                fieldnames=self.fieldnames
            )

            writer.writeheader()

            for row in self.metrics:
                writer.writerow(row)

        print(
            f"\n📊 Métricas salvas em:\n"
            f"{self.csv_path}"
        )

        return self.csv_path