"""
===============================================================================
strategies.py

Define as estratégias federadas usadas nos experimentos do CLARA-MEC.

Baselines implementadas:

1. FedSGD
2. FedAvg
3. FedProx
4. Personalized FL (PFL)
5. CLARA-MEC (FedAvg + Continual Learning)
6. FL + Heuristic Offloading

Autor: Nelson Machado Junior
===============================================================================
"""

import flwr as fl

# =============================================================================
# Aggregation function for evaluation metrics
# =============================================================================

def aggregate_metrics(metrics):

    accuracies = []
    latencies = []
    energies = []
    costs = []

    for num_examples, m in metrics:

        if "accuracy" in m:
            accuracies.append(m["accuracy"])

        if "latency" in m:
            latencies.append(m["latency"])

        if "energy" in m:
            energies.append(m["energy"])

        if "cost" in m:
            costs.append(m["cost"])

    results = {}

    if accuracies:
        results["accuracy"] = sum(accuracies) / len(accuracies)

    if latencies:
        results["latency"] = sum(latencies) / len(latencies)

    if energies:
        results["energy"] = sum(energies) / len(energies)

    if costs:
        results["cost"] = sum(costs) / len(costs)

    return results

# =============================================================================
# FedAvg (Clientes treinam várias épocas locais)
# =============================================================================
def create_fedavg_strategy(config, metrics_logger):

    return fl.server.strategy.FedAvg(
        fraction_fit=config["federated_learning"]["client_fraction"], # x% de total_clients participam por rodada
        fraction_evaluate=1.0, # Todos avaliam o modelo global
        min_fit_clients=max(
            1,
            int(
                config["federated_learning"]["client_fraction"]
                * config["federated_learning"]["total_clients"]
            )
        ), # Número mínimo de clientes de treino
        min_evaluate_clients=config["federated_learning"]["total_clients"],
        min_available_clients=config["federated_learning"]["total_clients"],
        on_fit_config_fn=lambda rnd: {"round": rnd}, # Config repassada aos clientes
        
        # métricas de treino
        # fit_metrics_aggregation_fn = metrics_logger.aggregate_fit_metrics,
        # fit_metrics_aggregation_fn=lambda x: {}, # Caso queira remover o warning "No fit_metrics_aggregation_fn provided" na simulação

        # métricas de avaliação
        evaluate_metrics_aggregation_fn = metrics_logger.aggregate_metrics
    )


# =============================================================================
# FedSGD (Sem treinamento local profundo)
# =============================================================================
def create_fedsgd_strategy(config, metrics_logger):

    return fl.server.strategy.FedAvg(
        fraction_fit=1.0, # Todos os clientes participam de cada rodada
        fraction_evaluate=1.0, # Todos avaliam o modelo global
        min_fit_clients=max(
            1,
            int(
                config["federated_learning"]["client_fraction"]
                * config["federated_learning"]["total_clients"]
            )
        ), # Número mínimo de clientes de treino
        min_evaluate_clients=config["federated_learning"]["total_clients"],
        min_available_clients=config["federated_learning"]["total_clients"],
        on_fit_config_fn=lambda rnd: { # Config repassada aos clientes
            "round": rnd,
            "local_epochs": 1,
            # "batch_size": config["federated_learning"]["batch_size"]
            "batch_size": 1
        },

        # métricas de treino
        # fit_metrics_aggregation_fn = metrics_logger.aggregate_fit_metrics,

        # métricas de avaliação
        evaluate_metrics_aggregation_fn = metrics_logger.aggregate_metrics
    )


# =============================================================================
# FedProx
# =============================================================================
def create_fedprox_strategy(config, metrics_logger):

    return fl.server.strategy.FedProx(
        fraction_fit=config["federated_learning"]["client_fraction"], # x% de total_clients participam por rodada
        fraction_evaluate=1.0, # Todos avaliam o modelo global
        min_fit_clients=max(
            1,
            int(
                config["federated_learning"]["client_fraction"]
                * config["federated_learning"]["total_clients"]
            )
        ), # Número mínimo de clientes de treino
        min_evaluate_clients=config["federated_learning"]["total_clients"],
        min_available_clients=config["federated_learning"]["total_clients"],
        proximal_mu=config["federated_learning"].get("proximal_mu", 0.01),
        on_fit_config_fn=lambda rnd: {"round": rnd}, # Config repassada aos clientes
        
        # métricas de treino
        # fit_metrics_aggregation_fn = metrics_logger.aggregate_fit_metrics,

        # métricas de avaliação
        evaluate_metrics_aggregation_fn = metrics_logger.aggregate_metrics
    )


# =============================================================================
# Personalized Federated Learning (PFL)
# =============================================================================
def create_pfl_strategy(config, metrics_logger):

    return fl.server.strategy.FedAvg(
        fraction_fit=config["federated_learning"]["client_fraction"], # x% de total_clients participam por rodada
        fraction_evaluate=1.0, # Todos avaliam o modelo global
        min_fit_clients=max(
            1,
            int(
                config["federated_learning"]["client_fraction"]
                * config["federated_learning"]["total_clients"]
            )
        ), # Número mínimo de clientes de treino
        min_evaluate_clients=config["federated_learning"]["total_clients"],
        min_available_clients=config["federated_learning"]["total_clients"],
        on_fit_config_fn=lambda rnd: { # Config repassada aos clientes
            "round": rnd,
            "personalized": True
        },
        
        # métricas de treino
        # fit_metrics_aggregation_fn = metrics_logger.aggregate_fit_metrics,

        # métricas de avaliação
        evaluate_metrics_aggregation_fn = metrics_logger.aggregate_metrics
    )


# =============================================================================
# FL + Heuristic (regra de offloading)
# =============================================================================
def create_fl_heuristic_strategy(config, metrics_logger):

    return fl.server.strategy.FedAvg(
        fraction_fit=config["federated_learning"]["client_fraction"], # x% de total_clients participam por rodada
        fraction_evaluate=1.0, # Todos avaliam o modelo global
        min_fit_clients=max(
            1,
            int(
                config["federated_learning"]["client_fraction"]
                * config["federated_learning"]["total_clients"]
            )
        ), # Número mínimo de clientes de treino
        min_evaluate_clients=config["federated_learning"]["total_clients"],
        min_available_clients=config["federated_learning"]["total_clients"],
        on_fit_config_fn=lambda rnd: { # Config repassada aos clientes
            "round": rnd,
            "heuristic_offloading": True
        },
        
        # métricas de treino
        # fit_metrics_aggregation_fn = metrics_logger.aggregate_fit_metrics,

        # métricas de avaliação
        evaluate_metrics_aggregation_fn = metrics_logger.aggregate_metrics
    )


# =============================================================================
# CLARA-MEC (baseline principal)
# =============================================================================
def create_claramec_strategy(config, metrics_logger):

    return fl.server.strategy.FedAvg(
        fraction_fit=config["federated_learning"]["client_fraction"], # x% de total_clients participam por rodada
        fraction_evaluate=1.0, # Todos avaliam o modelo global
        min_fit_clients=max(
            1,
            int(
                config["federated_learning"]["client_fraction"]
                * config["federated_learning"]["total_clients"]
            )
        ), # Número mínimo de clientes de treino
        min_evaluate_clients=config["federated_learning"]["total_clients"],
        min_available_clients=config["federated_learning"]["total_clients"],
        on_fit_config_fn=lambda rnd: { # Config repassada aos clientes
            "round": rnd,
            "continual_learning": True
        },
        
        # métricas de treino
        # fit_metrics_aggregation_fn = metrics_logger.aggregate_fit_metrics,

        # métricas de avaliação
        evaluate_metrics_aggregation_fn = metrics_logger.aggregate_metrics
    )

