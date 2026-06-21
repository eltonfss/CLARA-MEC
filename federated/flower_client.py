"""
===============================================================================
Arquivo: flower_client.py

Descrição:
Implementa o cliente federado do framework CLARA-MEC utilizando Flower.

Este cliente realiza:

- Recebimento do modelo global
- Treinamento local multitarefa
- Aplicação de Knowledge Distillation (KD)
- Aplicação de Elastic Weight Consolidation (EWC)
- Cálculo da loss total conforme Section III do artigo:

    L_total =
        L_task
        + lambda_KD * L_KD
        + lambda_EWC * L_EWC

- Interação com ambiente MEC (environment.py)
- Cálculo de latência, energia e recompensa        
- Envio dos pesos atualizados ao servidor

Autor: Nelson Machado Junior
===============================================================================
"""

import flwr as fl
import tensorflow as tf
import numpy as np

from federated.continual_learning import ContinualLearningManager
from mec.environment import MECEnvironment


# =============================================================================
# Classe FlowerClient
# =============================================================================
class FlowerClient(fl.client.NumPyClient):
    """
    Cliente federado responsável pelo treinamento local no CLARA-MEC.
    """

    # =========================================================================
    # Inicialização
    # =========================================================================    
    # ALTERAÇÃO #2
    def __init__(self, model, train_data, val_data, cid, config):
    # def __init__(self, model, cid, config):    
        """
        model           -> modelo multitarefa (multitask_policy_model)
        train_dataset   -> dataset de treino local
        val_dataset     -> dataset de validação local
        config          -> dicionário carregado do YAML
        """

        import random
        import numpy as np
        import tensorflow as tf

        seed = config["experiment"]["seed"] + int(cid)

        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        self.model = model
        self.x_train, self.y_train = train_data
        self.x_val, self.y_val = val_data
        # self.train_dataset = train_dataset
        # self.val_dataset = val_dataset
        self.cid = int(cid)
        self.config = config


        # Cria os datasets

        # self.train_dataset = tf.data.Dataset.from_tensor_slices(
        #     (self.x_train, self.y_train)
        # ).batch(config["federated_learning"]["batch_size"])

        # self.val_dataset = tf.data.Dataset.from_tensor_slices(
        #     (self.x_val, self.y_val)
        # ).batch(config["federated_learning"]["batch_size"])

        batch_size = config["federated_learning"]["batch_size"]

        self.train_dataset = (
            tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
            .shuffle(1000)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        self.val_dataset = (
            tf.data.Dataset.from_tensor_slices((self.x_val, self.y_val))
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
       
        # Carregar dados AQUI
        # from utils.data_loader import load_client_data
        # self.train_dataset, self.val_dataset = load_client_data(self.cid, config)

        # Inicializa o gerenciador de Continual Learning
        self.cl_manager = ContinualLearningManager(model, config)

        # Temperatura da KD
        self.temperature = config["continual_learning"]["temperature"]

        # Otimizador
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=config["training"]["learning_rate"]
        )

        # Número de épocas locais
        self.local_epochs = config["federated_learning"]["local_epochs"]

        # Ambiente MEC
        self.env = MECEnvironment(config)

        # Cria o teacher uma única vez (Cria cópia temporária do modelo global para gerar logits teacher)
        self.global_model = tf.keras.models.clone_model(self.model)
        self.global_model.set_weights(self.model.get_weights())

    # =========================================================================
    # Enviar pesos (parâmetros) ao servidor
    # =========================================================================
    def get_parameters(self, config):
        """
        Retorna os pesos atuais do modelo local.
        """
        weights = self.model.get_weights()

        # PFL: não enviar pesos das heads
        if config.get("personalized", False):
            return weights[:-2]

        return weights

    # =========================================================================
    # Receber pesos (parâmetros) do servidor
    # =========================================================================
    def set_parameters(self, parameters):
        """
        Atualiza o modelo local com os pesos globais.
        """
        current_weights = self.model.get_weights()

        # PFL → atualizar apenas backbone
        if len(parameters) < len(current_weights):
            updated_weights = parameters + current_weights[len(parameters):]
            self.model.set_weights(updated_weights)
        else:
            self.model.set_weights(parameters)

    # =========================================================================
    # Treinamento Local
    # =========================================================================
    def fit(self, parameters, config):
        """
        Executa o treinamento local em cada round federado.
        """        

        # Atualiza pesos com modelo global
        self.set_parameters(parameters)

        # # Armazena logits do modelo global para KD
        # global_model_weights = parameters
        # global_logits_cache = []

        # Cria cópia temporária do modelo global para gerar logits teacher
        # global_model = tf.keras.models.clone_model(self.model)
        # global_model.set_weights(parameters) # global_model.set_weights(global_model_weights)
        # Aqui, nessa versão anterior, clone model → milhares de vezes, ou seja, com 
        # 50 clientes × 100 rounds, isso cria 5000 clones de modelo. Muito pesado.
        
        # Atualiza teacher model (para KD) / Armazena logits do modelo global para KD
        self.global_model.set_weights(parameters)
        global_model = self.global_model

        strategy = self.config["experiment"]["strategy"]
        
        # Treinamento local
        for epoch in range(self.local_epochs):

            for x_batch, y_batch in self.train_dataset:

                with tf.GradientTape() as tape:

                    # Forward do modelo local (student)
                    off_logits, res_logits = self.model(x_batch, training=True)
                    
                    # Labels multitarefa
                    y_off = y_batch["offload_decision"]
                    y_res = y_batch["resource_allocation"]

                    # =========================================================
                    # Loss supervisionada multitarefa
                    # =========================================================
                    
                    y_off = tf.expand_dims(y_off, axis=-1)

                    loss_off = tf.reduce_mean(
                        tf.keras.losses.binary_crossentropy(
                            y_off, off_logits, from_logits=True
                        )
                    )

                    loss_res = tf.reduce_mean(
                        tf.keras.losses.sparse_categorical_crossentropy(
                            y_res, res_logits, from_logits=True
                        )
                    )

                    task_loss = loss_off + loss_res

                    # =========================================================
                    # Knowledge Distillation
                    # =========================================================
                    
                    global_off, global_res = global_model(
                        x_batch, training=False
                    )

                    kd_loss = (
                        self.cl_manager.compute_kd_loss(
                            off_logits, global_off, self.temperature
                        )
                        + self.cl_manager.compute_kd_loss(
                            res_logits, global_res, self.temperature
                        )
                    )

                    # =========================================================
                    # Elastic Weight Consolidation
                    # =========================================================
                        
                    ewc_loss = self.cl_manager.compute_ewc_loss()

                    # =========================================================
                    # Loss Total
                    # =========================================================
                    
                    total_loss = task_loss

                    # KD + EWC apenas no CLARA-MEC
                    if strategy == "claramec":  

                        total_loss += (
                            self.cl_manager.lambda_kd * kd_loss
                            + self.cl_manager.lambda_ewc * ewc_loss
                        )
                
                # Gradientes
                grads = tape.gradient(
                    total_loss, self.model.trainable_variables
                )

                # Atualização
                self.optimizer.apply_gradients(
                    zip(grads, self.model.trainable_variables)
                )

                # =============================================================
                # Interação com ambiente MEC
                # =============================================================
                use_heuristic = (strategy == "heuristic")

                if use_heuristic:

                    d_t = []
                    r_t = []

                    for state in x_batch:
                        task_size = state[0]
                        decision = 1 if task_size > 1e6 else 0
                        d_t.append(decision)
                        r_t.append(0)

                    d_t = tf.convert_to_tensor(d_t)
                    r_t = tf.convert_to_tensor(r_t)

                else:

                    d_t = tf.cast(tf.round(tf.sigmoid(off_logits)), tf.int32)
                    r_t = tf.argmax(res_logits, axis=1)

                for decision, resource in zip(d_t, r_t):

                    self.env.step((int(decision.numpy()), int(resource.numpy())))

        # =============================================================
        # Atualiza Fisher ao final do treinamento local
        # =============================================================

        # self.cl_manager.compute_fisher_information(self.train_dataset)
        self.cl_manager.compute_fisher_information(
            self.x_train,
            self.y_train["offload_decision"],
            self.y_train["resource_allocation"]
        )

        num_examples = 0
        for x_batch, _ in self.train_dataset:
            num_examples += x_batch.shape[0]

        return self.get_parameters({}), num_examples, {}  # len(self.train_dataset) no lugar de num_examples

    # =========================================================================
    # Avaliação Local
    # =========================================================================
    def evaluate(self, parameters, config):

        self.set_parameters(parameters)

        total_loss = 0.0
        total_samples = 0

        total_latency = 0.0
        total_energy = 0.0

        correct = 0
        total_examples = 0  # 🔥 CORREÇÃO

        strategy = self.config["experiment"]["strategy"]
        use_heuristic = (strategy == "heuristic")

        for x_batch, y_batch in self.val_dataset:

            off_logits, res_logits = self.model(x_batch, training=False)

            y_off = y_batch["offload_decision"]
            y_res = y_batch["resource_allocation"]

            # =========================
            # Losses
            # =========================

            y_off = tf.expand_dims(y_off, axis=-1)

            loss_off = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(
                    y_off, off_logits, from_logits=True
                )
            )

            loss_res = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    y_res, res_logits, from_logits=True
                )
            )

            total_loss += (loss_off + loss_res)
            total_samples += 1

            # =========================
            # Predições
            # =========================
            if use_heuristic:

                d_t = []
                r_t = []

                for state in x_batch:
                    task_size = state[0]
                    decision = 1 if task_size > 1e6 else 0
                    d_t.append(decision)
                    r_t.append(0)

                d_t = tf.convert_to_tensor(d_t)
                r_t = tf.convert_to_tensor(r_t)

            else:

                d_t = tf.cast(tf.round(tf.sigmoid(off_logits)), tf.int32)
                r_t = tf.argmax(res_logits, axis=1)

            for decision, resource in zip(d_t, r_t):

                _, _, _, info = self.env.step(
                    (int(decision.numpy()), int(resource.numpy()))
                )

                total_latency += info["latency"]
                total_energy += info["energy_consumed"]

            # ==================================================
            # Accuracy da decisão de offloading
            # ==================================================
            # pred = tf.cast(tf.round(tf.sigmoid(off_logits)), tf.float32)
            pred = tf.cast(tf.sigmoid(off_logits) > 0.5, tf.float32) 
            # Essa correção no pred se dá por conta de um pequeno erro conceitual
            #   na forma como se mede a accuracy de offloading que pode distorcer 
            #   as curvas de convergência. Ele aparece nesta linha:
            #   pred = tf.cast(tf.round(tf.sigmoid(off_logits)), tf.float32)
            #   Ele introduz threshold fixo 0.5, que nem sempre é ideal. 
            # A correção trocando-se esta linha por:
            #   pred = tf.cast(tf.sigmoid(off_logits) > 0.5, tf.float32)  
            #   gera curvas de convergência mais suaves, ou seja, 
            #   round():
            #   0.499 → 0
            #   0.501 → 1

            # correct += tf.reduce_sum(tf.cast(pred == y_off, tf.float32))
            correct += tf.reduce_sum(tf.cast(pred == y_off, tf.float32)).numpy()
            # O correct anterior retorna tensor. É mais seguro converter usando .numpy()
            #   para evitar mistura tensor/python (segurança numérica).

            batch_size = x_batch.shape[0]
            total_examples += batch_size

        avg_loss = float(total_loss / total_samples)

        accuracy = float(correct / total_examples)  

        avg_latency = total_latency / total_examples
        avg_energy = total_energy / total_examples

        alpha = self.config["cost_function"]["alpha_latency"]
        beta = self.config["cost_function"]["beta_energy"]

        avg_cost = alpha * avg_latency + beta * avg_energy

        return avg_loss, total_examples, {
            "loss": avg_loss,
            "accuracy": accuracy,
            "latency": avg_latency,
            "energy": avg_energy,
            "cost": avg_cost
        }