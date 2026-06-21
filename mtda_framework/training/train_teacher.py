"""
===============================================================================
train_teacher.py

Este é o treinamento no domínio origem

===============================================================================
"""

import tensorflow as tf
import numpy as np

from models.multitask_model import (
    build_multitask_model
)


def train_teacher(
    X_train,
    y_off,
    y_res,
    epochs=50,
    batch_size=64
):

    input_dim = X_train.shape[1]

    model = build_multitask_model(input_dim)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={
            "offload_decision":
                tf.keras.losses.BinaryCrossentropy(),

            "resource_allocation":
                tf.keras.losses.SparseCategoricalCrossentropy()
        },
        metrics={
            "offload_decision": ["accuracy"],
            "resource_allocation": ["accuracy"]
        }
    )

    history = model.fit(
        X_train,
        {
            "offload_decision": y_off,
            "resource_allocation": y_res
        },
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )

    return model, history