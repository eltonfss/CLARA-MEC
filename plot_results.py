"""
===============================================================================
plot_results.py

Geração automática de gráficos dos experimentos do CLARA-MEC.

Autor: Nelson Machado Junior
===============================================================================
"""
import glob
import pandas as pd
import matplotlib.pyplot as plt
import os
import yaml
from datetime import datetime

CONFIG_FILE = "config.yaml"

# Ler dataset diretamente do YAML
with open(CONFIG_FILE, "r") as f:
    config = yaml.safe_load(f)

dataset = config["dataset"]["name"]
baseline = config["experiment"]["name"]
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def plot_metric(df, metric, output_dir):

    plt.figure()

    plt.plot(df["round"], df[metric])

    plt.xlabel("Federated Rounds")
    plt.ylabel(metric.capitalize())

    plt.title(f"{metric.capitalize()} vs Rounds")

    plt.grid(True)

    output_path = os.path.join(
        output_dir,
        f"{metric}_{baseline}_{dataset}_{timestamp}.png"
    )

    plt.savefig(output_path)

    print(f"📈 Gráfico salvo: {output_path}")

    plt.close()


def generate_plots(csv_path):

    df = pd.read_csv(csv_path)

    output_dir = "results/plots"

    os.makedirs(output_dir, exist_ok=True)

    metrics = [
        "loss",
        "accuracy",
        "latency",
        "energy",
        "cost"
    ]

    for metric in metrics:

        if metric in df.columns:

            plot_metric(df, metric, output_dir)


# if __name__ == "__main__":

#     csv_path = input("Informe o caminho do CSV de métricas: ")

#     generate_plots(csv_path)


# ============================================================
# Localização automática do CSV
# ============================================================
if __name__ == "__main__":

    results_dir = "results"
    # PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
    # os.makedirs(PLOTS_DIR, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(results_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError("Nenhum arquivo CSV encontrado em 'results/'.")
    csv_path = csv_files[-1]
    print(f"📄 Usando arquivo de métricas: {csv_path}")

    generate_plots(csv_path)
