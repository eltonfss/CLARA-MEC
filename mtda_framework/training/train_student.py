# """
# ===============================================================================
# train_student.py

# Aqui está o MTDA propriamente dito.

# ===============================================================================
# """

# import tensorflow as tf
# import numpy as np

# from models.multitask_model import (
#     build_multitask_model
# )

# from continual.stochastic_recovery import (
#     stochastic_recovery
# )


# def distillation_loss(
#     teacher_logits,
#     student_logits,
#     temperature=2.0
# ):

#     teacher_probs = tf.nn.softmax(
#         teacher_logits / temperature
#     )

#     student_probs = tf.nn.softmax(
#         student_logits / temperature
#     )

#     loss = tf.keras.losses.KLDivergence()(
#         teacher_probs,
#         student_probs
#     )

#     return loss * (temperature ** 2)


# def train_student(
#     teacher_model,
#     X_target,
#     y_off,
#     y_res,
#     epochs=50,
#     lambda_kd=0.5,
#     recovery_rate=0.02
# ):

#     input_dim = X_target.shape[1]

#     student_model = build_multitask_model(
#         input_dim
#     )

#     optimizer = tf.keras.optimizers.Adam(
#         1e-3
#     )

#     dataset = tf.data.Dataset.from_tensor_slices(
#         (
#             X_target,
#             y_off,
#             y_res
#         )
#     )

#     dataset = dataset.batch(64)

#     history = {
#         "loss": [],
#         "accuracy": []
#     }

#     for epoch in range(epochs):

#         epoch_loss = []

#         correct = 0
#         total = 0

#         for x_batch, y_off_batch, y_res_batch in dataset:

#             with tf.GradientTape() as tape:

#                 # teacher prediction
#                 teacher_off, teacher_res = (
#                     teacher_model(
#                         x_batch,
#                         training=False
#                     )
#                 )

#                 # student prediction
#                 student_off, student_res = (
#                     student_model(
#                         x_batch,
#                         training=True
#                     )
#                 )

#                 # ==========================================
#                 # CORREÇÃO DE SHAPE DO OFFLOADING LABEL
#                 # ==========================================
#                 y_off_batch = tf.expand_dims(
#                     y_off_batch,
#                     axis=-1
#                 )

#                 # supervised losses
#                 loss_off = tf.reduce_mean(
#                     tf.keras.losses.binary_crossentropy(
#                         y_off_batch,
#                         student_off
#                     )
#                 )

#                 loss_res = tf.reduce_mean(
#                     tf.keras.losses.sparse_categorical_crossentropy(
#                         y_res_batch,
#                         student_res
#                     )
#                 )

#                 task_loss = (
#                     loss_off
#                     + loss_res
#                 )

#                 kd_loss = (
#                     distillation_loss(
#                         teacher_off,
#                         student_off
#                     )
#                     +
#                     distillation_loss(
#                         teacher_res,
#                         student_res
#                     )
#                 )

#                 total_loss = (
#                     task_loss
#                     +
#                     lambda_kd * kd_loss
#                 )

#             grads = tape.gradient(
#                 total_loss,
#                 student_model.trainable_variables
#             )

#             optimizer.apply_gradients(
#                 zip(
#                     grads,
#                     student_model.trainable_variables
#                 )
#             )

#             # recovery
#             stochastic_recovery(
#                 student_model,
#                 teacher_model,
#                 recovery_rate
#             )

#             pred = tf.round(student_off)

#             correct += tf.reduce_sum(
#                 tf.cast(
#                     pred == y_off_batch,
#                     tf.float32
#                 )
#             )

#             total += len(y_off_batch)

#             epoch_loss.append(
#                 total_loss.numpy()
#             )

#         acc = float(correct / total)

#         history["loss"].append(
#             np.mean(epoch_loss)
#         )

#         history["accuracy"].append(acc)

#         print(
#             f"Epoch {epoch+1} | "
#             f"loss={np.mean(epoch_loss):.4f} "
#             f"acc={acc:.4f}"
#         )

#     return student_model, history

"""
===============================================================================
train_student.py

Treinamento do Student Model do MTDA
(Implementação científica corrigida)

- Knowledge Distillation
- Stochastic Recovery (Eq. 14)
- Train/Test split
- Test Accuracy real (não treino)

===============================================================================
"""

import tensorflow as tf
import numpy as np

from sklearn.model_selection import (
    train_test_split
)

from models.multitask_model import (
    build_multitask_model
)

from continual.stochastic_recovery import (
    stochastic_recovery
)


# =============================================================================
# Distillation Loss
# =============================================================================
def distillation_loss(
    teacher_logits,
    student_logits,
    temperature=2.0
):

    teacher_probs = tf.nn.softmax(
        teacher_logits / temperature
    )

    student_probs = tf.nn.softmax(
        student_logits / temperature
    )

    loss = tf.keras.losses.KLDivergence()(
        teacher_probs,
        student_probs
    )

    return loss * (temperature ** 2)


# =============================================================================
# Student Training (MTDA)
# =============================================================================
def train_student(
    teacher_model,
    X_target,
    y_off,
    y_res,
    epochs=50,
    lambda_kd=0.5,
    recovery_rate=0.02
):

    # ==========================================================
    # Train / Test split (CORREÇÃO CIENTÍFICA)
    # ==========================================================
    (
        X_train,
        X_test,
        y_off_train,
        y_off_test,
        y_res_train,
        y_res_test
    ) = train_test_split(
        X_target,
        y_off,
        y_res,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    input_dim = X_train.shape[1]

    student_model = build_multitask_model(
        input_dim
    )

    optimizer = tf.keras.optimizers.Adam(
        # learning_rate=1e-3
        learning_rate=3e-4
    )

    # ==========================================================
    # Dataset de treino
    # ==========================================================
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            X_train,
            y_off_train,
            y_res_train
        )
    )

    dataset = dataset.batch(64)

    history = {
        "loss": [],
        "accuracy": []
    }

    # ==========================================================
    # Training Loop
    # ==========================================================
    for epoch in range(epochs):

        epoch_loss = []

        # ======================================================
        # TREINO
        # ======================================================
        for x_batch, y_off_batch, y_res_batch in dataset:

            with tf.GradientTape() as tape:

                # ==============================================
                # Teacher prediction
                # ==============================================
                teacher_off, teacher_res = (
                    teacher_model(
                        x_batch,
                        training=False
                    )
                )

                # ==============================================
                # Student prediction
                # ==============================================
                student_off, student_res = (
                    student_model(
                        x_batch,
                        training=True
                    )
                )

                # ==============================================
                # Shape correction
                # ==============================================
                y_off_batch = tf.expand_dims(
                    y_off_batch,
                    axis=-1
                )

                # ==============================================
                # Supervised losses
                # Eq. (7) + Eq. (8)
                # ==============================================
                loss_off = tf.reduce_mean(
                    tf.keras.losses.binary_crossentropy(
                        y_off_batch,
                        student_off
                    )
                )

                loss_res = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(
                        y_res_batch,
                        student_res
                    )
                )

                task_loss = (
                    loss_off
                    + loss_res
                )

                # ==============================================
                # Knowledge Distillation
                # ==============================================
                kd_loss = (
                    distillation_loss(
                        teacher_off,
                        student_off
                    )
                    +
                    distillation_loss(
                        teacher_res,
                        student_res
                    )
                )

                # ==============================================
                # Total Loss
                # ==============================================
                total_loss = (
                    task_loss
                    +
                    lambda_kd * kd_loss
                )

            # ==================================================
            # Gradients
            # ==================================================
            grads = tape.gradient(
                total_loss,
                student_model.trainable_variables
            )

            optimizer.apply_gradients(
                zip(
                    grads,
                    student_model.trainable_variables
                )
            )

            # ==================================================
            # Stochastic Recovery (Eq. 14)
            # ==================================================
            stochastic_recovery(
                student_model,
                teacher_model,
                recovery_rate
            )

            epoch_loss.append(
                total_loss.numpy()
            )

        # ======================================================
        # TEST ACCURACY (CORREÇÃO CIENTÍFICA)
        # ======================================================
        student_off_test, _ = student_model(
            X_test,
            training=False
        )

        pred_test = tf.cast(
            tf.round(
                tf.sigmoid(student_off_test)
            ),
            tf.float32
        )

        y_off_test_tensor = tf.expand_dims(
            y_off_test,
            axis=-1
        )

        test_acc = tf.reduce_mean(
            tf.cast(
                pred_test
                ==
                y_off_test_tensor,
                tf.float32
            )
        )

        avg_loss = float(
            np.mean(epoch_loss)
        )

        test_acc = float(
            test_acc.numpy()
        )

        history["loss"].append(
            avg_loss
        )

        history["accuracy"].append(
            test_acc
        )

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"loss={avg_loss:.4f} | "
            f"test_acc={test_acc:.4f}"
        )

    return student_model, history