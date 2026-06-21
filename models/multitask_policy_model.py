"""
================================================================================
Arquivo: multitask_policy_model.py
Descrição:
Implementa o modelo multi-tarefa do CLARA-MEC como um
Deep Neural Network Policy Model.

Preserva integralmente a estrutura original do PFCL-MEC:
- Backbone convolucional - CNN (Conv2D + MaxPooling + Flatten)
- Camadas densas compartilhadas
- n_actions
- Método compile()
- Carregamento de configurações via YAML

Ajustes realizados no CLARA-MEC para alinhamento com a Section III do artigo Wiley:
- Heads retornam logits puros (sem ativação final (softmax/sigmoid), ou seja, activation=None)
- Separação explícita entre logits de offloading e resource allocation com from_logits=True nas losses
- Compatibilidade total com Knowledge Distillation (KD)
- Compatibilidade com EWC

Autor: Nelson Machado Junior
================================================================================
"""

import tensorflow as tf
layers = tf.keras.layers
models = tf.keras.models


class MultiTaskPolicyModel:
    """
    Wrapper (classe) responsável por:
    - carregar config
    - consruir modelo
    - compilar
    - gerenciar pesos
    """
    def __init__(self, config):
        """
        Inicializa o modelo a partir das configurações do YAML.

        Parâmetros:
        - config: dicionário carregado do arquivo YAML
        """

        self.config = config
        self.input_shape = tuple(config["model"]["input_shape"])
        self.n_actions = config["model"]["heads"]["resource_allocation"]["n_actions"]
        self.learning_rate = config["training"]["learning_rate"]

        # Melhoria científica para evitar overfitting
        self.dropout_rate = config["model"]["backbone"]["dropout"]

        # Construção do modelo
        self.model = self._build_model()

        # Compilação do modelo
        self._compile_model()

    # =========================================================================
    # CONSTRUÇÃO DO MODELO - BACKBONE CNN (Feature Extractor REAL)
    # =========================================================================
    def _build_model(self):

        inputs = layers.Input(shape=self.input_shape)

        # =========================
        # BACKBONE CONVOLUCIONAL
        # =========================
        x = layers.Conv2D(32, (3, 3), activation="relu")(inputs)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(64, (3, 3), activation="relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(128, (3, 3), activation="relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)

        # Flatten
        x = layers.Flatten()(x)

        # =========================
        # CAMADAS DENSAS COMPARTILHADAS
        # =========================
        x = layers.Dense(256, activation="relu")(x)

        # Melhoria científica para evitar overfitting
        x = layers.Dropout(self.dropout_rate)(x)

        x = layers.Dense(128, activation="relu")(x)

        # =========================
        # HEAD 1: OFFLOADING DECISION
        # Sem ativação final → Logits puros (necessário para KD)
        # =========================
        offloading_logits = layers.Dense(
            1,
            activation=None,
            name="offloading_logits"
        )(x)

        # =========================
        # HEAD 2: RESOURCE ALLOCATION
        # n_actions preservado
        # Sem ativação final → Logits puros (necessário para KD)
        # =========================
        resource_logits = layers.Dense(
            self.n_actions,
            activation=None,
            name="resource_logits"
        )(x)

        model = models.Model(
            inputs=inputs,
            outputs=[offloading_logits, resource_logits]
        )

        return model

    # =========================================================================
    # COMPILAÇÃO DO MODELO
    # =========================================================================
    def _compile_model(self):
        """
        Compila o modelo.
        A loss total será controlada externamente no continual_learning.py.
        """

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate
        )

		# Losses aqui são placeholders.
        # As losses reais do CLARA-MEC são combinadas externamente.
        # Aqui mantemos compatibilidade com Keras.
        self.model.compile(
            optimizer=optimizer,
            loss={
                "offloading_logits": tf.keras.losses.BinaryCrossentropy(from_logits=True),
                "resource_logits": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            }
        )

    def __call__(self, inputs, training=False):
        """
        Torna o modelo callable. Essa linha transforma o wrapper (classe) MultiTaskPolicyModel
        em um objeto callable, ou seja: self.model(x_batch) funciona normalmente. Logo, com o wrapper
        callable, o código off_logits, res_logits = self.model(x_batch, training=True) no
        flower_client.py funciona exatamente como um tf.keras.Model. O ideal sem esse call seria 
        "model = create_multitask_model(config)" de "def create_multitask_model(config):" 
        e não "model = MultiTaskPolicyModel" de "class MultiTaskPolicyModel:" já que o método 
        get_model() quase nunca será usado porque FlowerClient normalmente recebe o modelo Keras
        diretamente. Como preferi manter a estrutura original não removendo o wrapper, a correção foi 
        manter o wrapper, mas tornar a classe utilizável diretamente como um modelo com esse call, 
        transformando o wrapper em um objeto callable.
        """        
        return self.model(inputs, training=training)

    # =========================================================================
    # MÉTODOS AUXILIARES
    # =========================================================================
    def get_model(self): 
        """Retorna o modelo Keras."""   	
        return self.model

    def get_weights(self): 
        """Retorna os pesos do modelo."""   	
        return self.model.get_weights()

    def set_weights(self, weights): 
        """Define os pesos do modelo."""    	
        self.model.set_weights(weights)

    def save(self, path): 
        """Salva o modelo."""    	
        self.model.save(path)

    def load(self, path): 
        """Carrega um modelo salvo."""   	
        self.model = tf.keras.models.load_model(path)