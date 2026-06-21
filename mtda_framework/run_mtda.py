"""
===============================================================================
run_mtda.py

Runner principal do MTDA original do artigo.

Fluxo:
1. Carrega config
2. Gera dataset sintético MEC
3. Treina Teacher Model
4. Treina Student Model (MTDA)
5. Salva métricas científicas CSV
===============================================================================
"""

from utils.config_loader import load_config
from utils.seed import set_global_seed

from train_mtda import train_mtda
from utils.metrics_logger import MetricsLogger

from plot_metrics import plot_metrics

def main():

    print("\n🚀 Iniciando MTDA...\n")

    # ==========================================================
    # Configuração
    # ==========================================================
    config = load_config()

    set_global_seed(
        config["experiment"]["seed"]
    )

    # ==========================================================
    # Logger
    # ==========================================================
    metrics_logger = MetricsLogger()

    # ==========================================================
    # Executa treinamento completo MTDA
    # ==========================================================
    history = train_mtda(config)  

    logger = MetricsLogger()

    csv_path = logger.save(history)

    plot_metrics(csv_path)

    # ==========================================================
    # Salvar métricas por epoch
    # ==========================================================
    for epoch in range(len(history["loss"])):

        metrics_logger.log(
            epoch=epoch + 1,

            train_loss=float(
                history["loss"][epoch]
            ),

            # ainda não há teste separado
            test_loss=float(
                history["loss"][epoch]
            ),

            accuracy_offloading=float(
                history["accuracy"][epoch]
            ),

            # Sprint 5
            accuracy_resource=0.0,

            # Sprint 5
            latency=0.0,
            energy=0.0,
            cost=0.0,

            # Sprint 5
            forgetting_score=0.0,
            domain_gap=0.0
        )

    # ==========================================================
    # Salva CSV
    # ==========================================================
    metrics_logger.save()

    print("\n✅ Simulação finalizada.\n")


if __name__ == "__main__":
    main()