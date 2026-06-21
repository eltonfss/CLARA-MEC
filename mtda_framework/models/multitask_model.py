# """
# Multi-task neural network
# Article: Multi-task Domain Adaptation
# """

# import tensorflow as tf


# def build_multitask_model(input_dim):

#     inputs = tf.keras.Input(shape=(input_dim,))

#     x = tf.keras.layers.Dense(128, activation="relu")(inputs)
#     x = tf.keras.layers.BatchNormalization()(x)

#     x = tf.keras.layers.Dense(64, activation="relu")(x)
#     x = tf.keras.layers.Dropout(0.2)(x)

#     shared = tf.keras.layers.Dense(
#         32,
#         activation="relu"
#     )(x)

#     # Task 1: offloading decision
#     offload_output = tf.keras.layers.Dense(
#         1,
#         activation="sigmoid",
#         name="offload_decision"
#     )(shared)

#     # Task 2: resource allocation
#     resource_output = tf.keras.layers.Dense(
#         3,
#         activation="softmax",
#         name="resource_allocation"
#     )(shared)

#     model = tf.keras.Model(
#         inputs=inputs,
#         outputs=[
#             offload_output,
#             resource_output
#         ]
#     )

#     return model

"""
Multi-task neural network
MTDA - versão científica
"""

import tensorflow as tf


def build_multitask_model(input_dim):

    inputs = tf.keras.Input(
        shape=(input_dim,)
    )

    x = tf.keras.layers.Dense(
        64,
        activation="relu"
    )(inputs)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dropout(
        0.35
    )(x)

    x = tf.keras.layers.Dense(
        32,
        activation="relu"
    )(x)

    x = tf.keras.layers.Dropout(
        0.25
    )(x)

    shared = tf.keras.layers.Dense(
        16,
        activation="relu"
    )(x)

    offload_output = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="offload_decision"
    )(shared)

    resource_output = tf.keras.layers.Dense(
        3,
        activation="softmax",
        name="resource_allocation"
    )(shared)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=[
            offload_output,
            resource_output
        ]
    )

    return model