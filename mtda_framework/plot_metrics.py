"""
===============================================================================
plot_metrics.py

Gera gráficos científicos do MTDA.

===============================================================================
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_metrics(csv_path):

    df = pd.read_csv(csv_path)

    os.makedirs(
        "results/plots",
        exist_ok=True
    )

    # ===================================
    # Accuracy Offloading
    # ===================================
    plt.figure(figsize=(8, 5))

    plt.plot(
        df["epoch"],
        # df["accuracy"],
        df["accuracy_offloading"],
        marker="o"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(
        "MTDA Test Accuracy"
    )

    plt.grid(True)

    acc_path = (
        "results/plots/"
        "accuracy_curve.png"
    )

    plt.savefig(acc_path)

    plt.close()

    # ===================================
    # Train Loss
    # ===================================
    plt.figure(figsize=(8, 5))

    plt.plot(
        df["epoch"],
        # df["loss"],
        df["train_loss"],
        marker="o"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        "MTDA Training Loss"
    )

    plt.grid(True)

    loss_path = (
        "results/plots/"
        "loss_curve.png"
    )

    plt.savefig(loss_path)

    plt.close()

    print(
        "\n📈 Plots gerados:"
    )

    print(acc_path)
    print(loss_path)