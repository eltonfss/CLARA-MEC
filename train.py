"""
===============================================================================
train.py

Script principal de execução do CLARA-MEC.

Responsável por:

- carregar configurações do YAML
- preparar datasets federados
- inicializar modelo multitarefa
- criar clientes Flower
- executar treinamento federado
- salvar métricas do experimento

Autor: Nelson Machado Junior
===============================================================================
"""
# =============================================================================
# Configurações de ambiente (DEVEM vir antes de inicializar TensorFlow)
# =============================================================================

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"

# =============================================================================
# Imports
# =============================================================================

import random
import flwr as fl
import numpy as np
import yaml
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

# Limitar paralelismo do TensorFlow
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from models.multitask_policy_model import MultiTaskPolicyModel
from federated.flower_client import FlowerClient
from utils.data_loader import prepare_federated_data
from utils.metrics_logger import MetricsLogger
from federated.strategies import (
    create_fedavg_strategy,
    create_fedsgd_strategy,
    create_fedprox_strategy,
    create_pfl_strategy,
    create_fl_heuristic_strategy,
    create_claramec_strategy
)
from flwr.common import Context

# =============================================================================
# Carregar configuração YAML
# =============================================================================
def load_config(path="config.yaml"):

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config

# =============================================================================
# Criar função geradora de clientes
# =============================================================================
def create_client_fn(config, client_train_datasets, client_val_datasets):

    def client_fn(context: Context):

        raw_cid = int(context.node_id)
        cid = raw_cid % config["federated_learning"]["total_clients"]
        # cid = int(context.node_id) % len(client_train_datasets)

        model_wrapper = MultiTaskPolicyModel(config)

        train_data = client_train_datasets[cid]
        val_data = client_val_datasets[cid]

        client = FlowerClient(
            model_wrapper.get_model(),
            train_data,
            val_data,
            cid,
            config
        )

        return client.to_client()

    return client_fn

# =============================================================================
# Estratégia federada
# =============================================================================
def create_strategy(config, metrics_logger):

    strategy_name = config["experiment"]["strategy"].lower()

    if strategy_name == "fedavg":
        return create_fedavg_strategy(config, metrics_logger)

    elif strategy_name == "fedsgd":
        return create_fedsgd_strategy(config, metrics_logger)

    elif strategy_name == "fedprox":
        return create_fedprox_strategy(config, metrics_logger)

    elif strategy_name == "pfl":
        return create_pfl_strategy(config, metrics_logger)

    elif strategy_name == "heuristic":
        return create_fl_heuristic_strategy(config, metrics_logger)
    
    elif strategy_name == "claramec":
        return create_claramec_strategy(config, metrics_logger)

    else:
        raise ValueError(f"Estratégia desconhecida: {strategy_name}")

# =============================================================================
# Função principal
# =============================================================================
def main():

    print("\n🚀 Iniciando CLARA-MEC...\n")

    # ---------------------------------------------------------
    # carregar config
    # ---------------------------------------------------------

    from utils.seed import set_global_seed
    config = load_config()

    # rounds = config["experiment"]["rounds"]
    num_clients = config["federated_learning"]["total_clients"]
    set_global_seed(config["experiment"]["seed"])

    # ---------------------------------------------------------
    # preparar dados federados
    # ---------------------------------------------------------

    print("📦 Carregando datasets...")

    # clients_data = prepare_federated_data(config)
    client_train_datasets, client_val_datasets = prepare_federated_data(config)
    print("CLIENT TRAIN DATASETS:", len(client_train_datasets))

    # ---------------------------------------------------------
    # logger de métricas
    # ---------------------------------------------------------

    metrics_logger = MetricsLogger()

    # ---------------------------------------------------------
    # estratégia federada
    # ---------------------------------------------------------

    strategy = create_strategy(config, metrics_logger)

    # ---------------------------------------------------------
    # criar client_fn
    # ---------------------------------------------------------

    client_fn = create_client_fn(
        config,
        client_train_datasets,
        client_val_datasets
    )   
    # client_fn = create_client_fn(config) 

    # ---------------------------------------------------------
    # iniciar simulação Flower
    # ---------------------------------------------------------

    print("\n🌐 Iniciando treinamento federado...\n")

    # prepare_federated_data(config) 

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(
            num_rounds=config["experiment"]["rounds"]
        ),
        strategy=strategy,
        client_resources={"num_cpus": 1}, # Ray criará menos concorrência
        ray_init_args={"num_cpus": 2} # Limita o paralelismo do Ray, evitando que este crie muitos actors simulatâneos
    )

    # ---------------------------------------------------------
    # salvar métricas
    # ---------------------------------------------------------

    metrics_logger.save()

    print("\n✅ Treinamento finalizado.")
    print("📊 Métricas salvas em ./results/\n")

if __name__ == "__main__":
    main()