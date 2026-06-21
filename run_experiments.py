"""
===============================================================================
run_experiments.py

Executa automaticamente multiplos experimentos do CLARA-MEC.

Observação importante:
Após a alteração no main() para apenas chamar o dataset configurado no YAML
(um dataset por simulação) pode-se manter a função update_dataset(), porque 
ela continua útil se quiser rodar batches de experimentos depois, mas ela não 
será usada nesse modo simples.

Dica rápida:
Uma prática comum é permitir dois modos de execução:
1. modo simples (dataset do YAML)
    python run_experiments.py
2. modo batch (todos datasets)
    python run_experiments.py --all

Autor: Nelson Machado Junior
===============================================================================
"""

import subprocess
import yaml

from baselines.mtda_centralized import run_mtda_experiment
from utils.data_loader import prepare_federated_data

CONFIG_FILE = "config.yaml"

# =============================================================================
# Atualiza dataset no YAML
# =============================================================================
def update_dataset(dataset_name):

    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)

    config["dataset"]["name"] = dataset_name

    if dataset_name in ["MNIST", "FMNIST"]:
        config["model"]["input_shape"] = [28, 28, 1]
    else:
        config["model"]["input_shape"] = [32, 32, 3]

    with open(CONFIG_FILE, "w") as f:
        yaml.dump(config, f)

    return config


# =============================================================================
# Executa experimento
# =============================================================================
def run_experiment(dataset):

    print("\n====================================")
    print(f"Executando experimento: {dataset}")
    print("====================================\n")

    config = update_dataset(dataset)

    strategy = config["experiment"]["strategy"]

    # =========================================================
    # MTDA (NAO federado)
    # =========================================================
    if strategy == "mtda":

        print("Running MTDA centralized baseline...")

        dataset_data = prepare_federated_data(config)

        run_mtda_experiment(config, dataset_data)

    # =========================================================
    # Federated (Flower)
    # =========================================================
    else:

        subprocess.run(["python", "train.py"])


# =============================================================================
# Loop principal
# =============================================================================
# def main():

#     datasets = [
#         "CIFAR10",
#         "CIFAR100",
#         "MNIST",
#         "FMNIST"
#     ]

#     for dataset in datasets:
#         run_experiment(dataset)


# if __name__ == "__main__":
#     main()

def main():

    # Ler dataset diretamente do YAML
    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)

    dataset = config["dataset"]["name"]

    run_experiment(dataset)


if __name__ == "__main__":
    main()

