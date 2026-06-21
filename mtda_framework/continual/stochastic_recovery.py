"""
===============================================================================
stochastic_recovery.py

Implementa recovery estocástico simples mas fiel ao artigo do MTDA onde:

"update student model weights
to avoid catastrophic forgetting"

===============================================================================
"""

import tensorflow as tf
import numpy as np


def stochastic_recovery(
    student_model,
    teacher_model,
    recovery_rate=0.02
):

    teacher_weights = teacher_model.get_weights()
    student_weights = student_model.get_weights()

    recovered = []

    for sw, tw in zip(
        student_weights,
        teacher_weights
    ):

        noise = np.random.uniform(
            0,
            recovery_rate,
            sw.shape
        )

        new_w = (
            (1 - noise) * sw
            +
            noise * tw
        )

        recovered.append(new_w)

    student_model.set_weights(recovered)