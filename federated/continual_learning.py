"""
===============================================================================
Arquivo: continual_learning.py

Descrição:
Implementa os mecanismos de Continual Learning do framework CLARA-MEC,
incluindo:

- Knowledge Distillation (KD)
- Elastic Weight Consolidation (EWC)
- Cálculo da Fisher Information Matrix
- Monitoramento de Esquecimento (ForgettingTracker)

A loss total segue exatamente a formulação da Section III do artigo:

L_total^k =
    L_task^k
    + lambda_KD * L_KD^k
    + lambda_EWC * L_EWC^k

Autor: Nelson Machado Junior
===============================================================================
"""

import tensorflow as tf
import numpy as np


# =============================================================================
# Classe ContinualLearningManager
# =============================================================================
class ContinualLearningManager:
    """
    Gerencia:
    - Knowledge Distillation
    - Elastic Weight Consolidation (EWC)
    - Fisher Information Matrix
    """

    def __init__(self, model, config):
        self.model = model
        self.lambda_kd = config["continual_learning"]["lambda_kd"]
        self.lambda_ewc = config["continual_learning"]["lambda_ewc"]
        self.temperature = config["continual_learning"]["temperature"]

        # Armazenamento do modelo consolidado (theta_old)
        self.previous_weights = None

        # Fisher Information Matrix
        self.fisher_matrix = None

    # =========================================================================
    # Knowledge Distillation Loss
    # =========================================================================
    def compute_kd_loss(self, local_logits, global_logits, temperature):
        """
        Calcula KL(global || local) conforme definido no artigo.

        local_logits  -> saída do modelo local (student)
        global_logits -> saída do modelo global (teacher)
        """
        # Usa temperatura definida no YAML
        temperature = self.temperature
        
		# Softmax com temperatura
        local_soft = tf.nn.softmax(local_logits / temperature)
        global_soft = tf.nn.softmax(global_logits / temperature)

        kd_loss = tf.keras.losses.KLDivergence()(global_soft, local_soft)

        return kd_loss

    # =========================================================================
    # Cálculo da Fisher Information Matrix (MULTITAREFA CORRIGIDO)
    # =========================================================================
    def compute_fisher_information(self, x_train, y_train_off, y_train_res):
        """
        Estima a Fisher Information Matrix utilizando os gradientes da LOSS MULTITAREFA,
        protegendo tanto a head de offloading quanto a head de resource allocation e o 
        backbone compartilhado.

		Isso garante consistência com a formulação matemática da Section III do artigo para
		a Wiley, onde:
		
        L_task = L_offloading + L_resource
        """

        fisher = []

        for var in self.model.trainable_variables:
            fisher.append(tf.zeros_like(var))

        with tf.GradientTape() as tape:
            preds = self.model(x_train, training=True)

            # Separação explícita das duas heads
            off_logits = preds[0]
            res_logits = preds[1]

            # A perda (loss) para as duas heads já é média pois usamos tf.reduce_mean e o gradiente já é derivado da média
            # Logo, já estamos usando o gradiente médio
            
            y_train_off = tf.expand_dims(y_train_off, axis=-1)
            # Loss de offloading (binária com logits)     
            off_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(
                    y_train_off,
                    off_logits,
                    from_logits=True
                )
            )

            # Loss de resource allocation (categórica com logits)
            res_loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    y_train_res,
                    res_logits,
                    from_logits=True
                )
            )

			# LOSS MULTITAREFA TOTAL
            total_task_loss = off_loss + res_loss

        grads = tape.gradient(total_task_loss, self.model.trainable_variables)

        for i in range(len(grads)):
            fisher[i] = tf.square(grads[i])

		# Armazena Fisher e pesos consolidados
        self.fisher_matrix = fisher
        self.previous_weights = [
            tf.identity(var) for var in self.model.trainable_variables
        ]

    # =========================================================================
    # Elastic Weight Consolidation Loss
    # =========================================================================
    def compute_ewc_loss(self):
        """
        Implementa:

        L_EWC^k = sum_i F_i^k (theta_i^k - theta_i_old^k)^2
        """

        if self.fisher_matrix is None or self.previous_weights is None:
            return 0.0

        ewc_loss = 0.0

        for i, var in enumerate(self.model.trainable_variables):
            ewc_loss += tf.reduce_sum(
                self.fisher_matrix[i] *
                tf.square(var - self.previous_weights[i])
            )

        return ewc_loss

    # =========================================================================
    # Loss Total (Task + KD + EWC)
    # =========================================================================
    def compute_total_loss(
        self,
        task_loss,
        local_logits,
        global_logits
    ):
        """
        Calcula a loss total conforme Section III.
        """

        kd_loss = self.compute_kd_loss(local_logits, global_logits)
        ewc_loss = self.compute_ewc_loss()

        total_loss = (
            task_loss
            + self.lambda_kd * kd_loss
            + self.lambda_ewc * ewc_loss
        )

        return total_loss, kd_loss, ewc_loss


# =============================================================================
# Classe ForgettingTracker
# =============================================================================
class ForgettingTracker:
    """
    Monitora o esquecimento catastrófico ao longo dos rounds.
    """

    def __init__(self):
        self.best_accuracies = {}
        self.current_accuracies = {}

    def update(self, task_id, accuracy):
        """
        Atualiza métricas de desempenho para uma tarefa específica.
        """

        if task_id not in self.best_accuracies:
            self.best_accuracies[task_id] = accuracy

        self.current_accuracies[task_id] = accuracy

        if accuracy > self.best_accuracies[task_id]:
            self.best_accuracies[task_id] = accuracy

    def compute_forgetting(self):
        """
        Calcula o índice médio de esquecimento.
        """

        forgetting_values = []

        for task_id in self.best_accuracies:
            best = self.best_accuracies[task_id]
            current = self.current_accuracies.get(task_id, 0)
            forgetting_values.append(best - current)

        if len(forgetting_values) == 0:
            return 0.0

        return np.mean(forgetting_values)