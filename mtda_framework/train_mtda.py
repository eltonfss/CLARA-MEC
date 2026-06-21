"""
===============================================================================
train_mtda.py

Pipeline principal do MTDA original do artigo.

Fluxo:
1. Gerar domínio origem e destino
2. Resolver MINLP via GEKKO
3. Treinar Teacher Model
4. Treinar Student Model (MTDA)
5. Retornar history para CSV
===============================================================================
"""

import numpy as np

from data.mec_generator import MECGenerator

from optimization.gekko_solver import (
    GekkkoMECOptimizer
)

from training.train_teacher import (
    train_teacher
)

from training.train_student import (
    train_student
)


def train_mtda(config):

    epochs = config["experiment"]["epochs"]

    print("\n📦 Gerando domínios MEC...\n")

    # ==========================================================
    # Generate domains
    # ==========================================================
    generator = MECGenerator(config)

    source_domain = (
        generator.generate_source_domain()
    )

    target_domain = (
        generator.generate_target_domain()
    )

    print(
        f"✔ Source Domain: "
        f"{source_domain.shape}"
    )

    print(
        f"✔ Target Domain: "
        f"{target_domain.shape}"
    )

    # ==========================================================
    # Solve MINLP with GEKKO
    # ==========================================================
    print("\n⚙ Resolvendo MINLP via GEKKO...\n")

    optimizer = (
        GekkkoMECOptimizer(config)
    )

    source_labels = (
        optimizer.solve_dataset(
            source_domain
        )
    )

    target_labels = (
        optimizer.solve_dataset(
            target_domain
        )
    )

    print(
        "✔ Source labels:",
        source_labels.keys()
    )

    print(
        "✔ Target labels:",
        target_labels.keys()
    )

    # ==========================================================
    # Prepare datasets
    # ==========================================================
    print("\n📦 Preparando datasets...\n")

    Xs = np.asarray(
        source_domain,
        dtype=np.float32
    )

    Xt = np.asarray(
        target_domain,
        dtype=np.float32
    )

    y_off_s = np.asarray(
        source_labels["offload_decision"],
        dtype=np.float32
    )

    y_res_s = np.asarray(
        source_labels["resource_allocation"],
        dtype=np.int32
    )

    y_off_t = np.asarray(
        target_labels["offload_decision"],
        dtype=np.float32
    )

    y_res_t = np.asarray(
        target_labels["resource_allocation"],
        dtype=np.int32
    )

    print(
        f"✔ Xs shape: {Xs.shape}"
    )

    print(
        f"✔ Xt shape: {Xt.shape}"
    )

    # ==========================================================
    # Train Teacher
    # ==========================================================
    print("\n🎓 Treinando Teacher Model...\n")

    teacher_model, _ = train_teacher(
        X_train=Xs,
        y_off=y_off_s,
        y_res=y_res_s,
        epochs=epochs,
        batch_size=64
    )

    print("\n✔ Teacher treinado.\n")

    # ==========================================================
    # Train Student (MTDA)
    # ==========================================================
    print(
        "\n🧠 Treinando Student Model "
        "(MTDA)...\n"
    )

    student_model, history = train_student(
        teacher_model=teacher_model,

        X_target=Xt,

        y_off=y_off_t,
        y_res=y_res_t,

        epochs=epochs,

        lambda_kd=config[
            "continual_learning"
        ]["lambda_kd"],

        recovery_rate=0.02
    )

    print("\n✔ Student treinado.\n")

    # ==========================================================
    # Return history for CSV
    # ==========================================================
    return history