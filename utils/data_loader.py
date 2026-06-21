"""
===============================================================================
Arquivo: data_loader.py

Descrição:
Responsável por carregar datasets reais e preparar os dados para
treinamento simulado de task offloading em ambiente federado no 
framework CLARA-MEC.

Datasets suportados:
- MNIST
- Fashion-MNIST (FMNIST)
- CIFAR-10
- CIFAR-100

Funcionalidades:

1) Carregamento de datasets reais
2) Normalização dos dados
3) Conversão para formato compatível com CNN
4) Geração de saídas multitarefa:
   - offload_decision
   - resource_allocation
5) Divisão federada entre clientes
6) Suporte para:
   - IID split
   - Non-IID split

Este módulo é utilizado diretamente pelo pipeline de treinamento
(train.py) e pelos clientes federados do Flower.

Autor: Nelson Machado Junior
===============================================================================
"""

import numpy as np
import tensorflow as tf

# =============================================================================
# DATASETS GLOBAIS (Ray-safe)
# =============================================================================

client_train_datasets = None
client_val_datasets = None
test_dataset = None


# =============================================================================
# CARREGAMENTO DE DATASETS
# =============================================================================
def load_dataset(dataset_name="CIFAR10"):
    """
    Carrega datasets reais disponíveis no TensorFlow/Keras.

    Parâmetros
    ----------
    dataset_name : str
        Nome do dataset a ser carregado.

    Retorna
    -------
    (x_train, y_train), (x_test, y_test)
    """
    dataset_name = dataset_name.upper()

    # -------------------------------------------------------------------------
    # Carregamento e normalização por dataset
    #
    # O trecho do código: 
    #   (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # retorna tensores de forma (50000, 32, 32, 3), onde:
    # - 50000 → número de amostras,
    # - 32, 32, 3 → shape da entrada da rede (altura, largura, canais).
    #
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # MNIST
    # -------------------------------------------------------------------------
    if dataset_name == "MNIST":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.astype("float32") / 255.0  # Normalização [0, 1]
        x_test = x_test.astype("float32") / 255.0    # Normalização [0, 1]

        # CNN exige dimensão de canal
        # Adicionam uma nova dimensão no final do tensor, criando um canal explícito (necessário para CNNs).
        # Datasets como MNIST e FMNIST são originalmente 2D (28×28) — ou seja, cada imagem é uma matriz 
        #   de pixels em escala de cinza, sem canal explícito. Mas as redes convolucionais (CNNs) do TensorFlow/Keras 
        #   esperam uma entrada 3D no formato: (altura, largura, canais). Para MNIST, isso deve ser: (28, 28, 1).
		# O último número (1) indica o canal de cor (preto e branco = 1 canal).
        x_train = np.expand_dims(x_train, -1)        # CNN espera canal explícito → (28,28,1)
        x_test = np.expand_dims(x_test, -1)          # CNN espera canal explícito → (28,28,1)

    # -------------------------------------------------------------------------
    # Fashion-MNIST
    # -------------------------------------------------------------------------
    elif dataset_name == "FMNIST":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        x_train = x_train.astype("float32") / 255.0  # Normalização [0, 1]
        x_test = x_test.astype("float32") / 255.0    # Normalização [0, 1]

        # CNN exige dimensão de canal
        # Adicionam uma nova dimensão no final do tensor, criando um canal explícito (necessário para CNNs).
        # Datasets como MNIST e FMNIST são originalmente 2D (28×28) — ou seja, cada imagem é uma matriz 
        #   de pixels em escala de cinza, sem canal explícito. Mas as redes convolucionais (CNNs) do TensorFlow/Keras 
        #   esperam uma entrada 3D no formato: (altura, largura, canais). Para MNIST, isso deve ser: (28, 28, 1).
		# O último número (1) indica o canal de cor (preto e branco = 1 canal).
        x_train = np.expand_dims(x_train, -1)        # CNN espera canal explícito → (28,28,1)
        x_test = np.expand_dims(x_test, -1)          # CNN espera canal explícito → (28,28,1)

    # -------------------------------------------------------------------------
    # CIFAR10
    # -------------------------------------------------------------------------
    elif dataset_name == "CIFAR10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data() # CIFAR10 já possui formato (32,32,3) → mantido para CNN
        x_train = x_train.astype("float32") / 255.0  # Normalização [0, 1]
        x_test = x_test.astype("float32") / 255.0    # Normalização [0, 1]

        # O flatten não achata as imagens, e sim os rótulos (labels), isto é, as saídas de classe.
        # Os rótulos do CIFAR10/CIFAR100 vêm com forma 2D (N, 1) — por exemplo:
        # y_train.shape = (50000, 1)
		# Mas as redes Keras esperam um vetor 1D: y_train.shape = (50000,)
		# Logo, o flatten() é apenas para transformar (N, 1) → (N,).
        y_train = y_train.flatten()   # flatten apenas nos rótulos, pois vêm em (N,1)
        y_test = y_test.flatten()     # flatten apenas nos rótulos, pois vêm em (N,1)

    # -------------------------------------------------------------------------
    # CIFAR100
    # -------------------------------------------------------------------------
    elif dataset_name == "CIFAR100":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode="fine") # CIFAR100 já possui formato (32,32,3) → mantido para CNN
        x_train = x_train.astype("float32") / 255.0  # Normalização [0, 1]
        x_test = x_test.astype("float32") / 255.0    # Normalização [0, 1]

        # O flatten não achata as imagens, e sim os rótulos (labels), isto é, as saídas de classe.
        # Os rótulos do CIFAR10/CIFAR100 vêm com forma 2D (N, 1) — por exemplo:
        # y_train.shape = (50000, 1)
		# Mas as redes Keras esperam um vetor 1D: y_train.shape = (50000,)
		# Logo, o flatten() é apenas para transformar (N, 1) → (N,).
        y_train = y_train.flatten()   # flatten apenas nos rótulos, pois vêm em (N,1)
        y_test = y_test.flatten()     # flatten apenas nos rótulos, pois vêm em (N,1)

    else:

        raise ValueError(f"Dataset {dataset_name} não suportado.")

    return (x_train, y_train), (x_test, y_test)


# =============================================================================
# GERAÇÃO DE SAÍDAS MULTITAREFA (MULTITASK LABELS)
# =============================================================================
def create_multitask_labels(y, n_actions):
    """
    Converte rótulos de classificação em saídas multitarefa.

    Task 1:
        Offloading decision (binário)

    Task 2:
        Resource allocation (classificação)

    Parâmetros
    ----------
    y : array
        Labels originais do dataset

    n_actions : int
        Número de políticas de alocação de recursos

    Retorna
    -------
    dict com labels multitarefa
    """
    # -------------------------------------------------------------
    # Offloading decision
    # -------------------------------------------------------------
    offload_decision = (y % 2).astype(np.float32)

    # -------------------------------------------------------------
    # Resource allocation
    # -------------------------------------------------------------
    resource_allocation = (y % n_actions).astype(np.int32)

    return {
        "offload_decision": offload_decision,
        "resource_allocation": resource_allocation
    }


# =============================================================================
# SPLIT IID
# =============================================================================
def split_iid(x, y_dict, num_clients):
    """
    Divide dados de forma IID entre clientes federados.
    Cada cliente recebe dados aleatórios da mesma distribuição.

    :param x: dados de entrada
    :param y_off: saída (rótulos) binária (offload decision) - classificação binária
    :param y_res: saída (rótulos) contínua (resource allocation) - regressão (linear) contínua
    :param num_clients: número de clientes federados
    :return: lista de tuplas (x_i, y_off_i, y_res_i)
    """
    # Divide o dataset igualmente entre os clientes
    data_per_client = len(x) // num_clients

    indices = np.random.permutation(len(x))

    x = x[indices]

    y_off = y_dict["offload_decision"][indices]
    y_res = y_dict["resource_allocation"][indices]

    clients = []

    for i in range(num_clients):
        start = i * data_per_client
        end = (i + 1) * data_per_client
        clients.append((
            x[start:end],
            {
                "offload_decision": y_off[start:end],
                "resource_allocation": y_res[start:end]
            }
        ))

    return clients


# =============================================================================
# SPLIT NON-IID
# =============================================================================
def split_noniid(x, y_dict, num_clients):
    """
    Divide dados de forma não-IID entre clientes federados.
    Cada cliente recebe subconjuntos distintos não-IID (não idênticos),
    ou seja, subconjuntos com distribuição diferente de rótulos do dataset 
    para simular heterogeneidade entre dispositivos MEC.

    :param x: dados de entrada
    :param y_off: saída (rótulos) binária (offload decision) - classificação binária
    :param y_res: saída (rótulos) contínua (resource allocation) - regressão (linear) contínua
    :param num_clients: número de clientes federados
    :return: lista de tuplas (x_i, y_off_i, y_res_i)
    """
    # Ordena por rótulo de decisão para gerar distribuição não-IID
    # Isso garante que cada cliente tenha distribuições diferentes (não-IID)
    # Exemplo:
    # - Cliente A pode processar tarefas relacionadas a imagens de "carros" (classe 1)
    # - Cliente B, imagens de "aviões" (classe 2)
    # - Cliente C, "cães" e "gatos" (classes 3 e 4)
    # Essa não uniformidade é o que caracteriza dados não-IID. Sem o np.argsort(y_off), os dados são distribuídos 
    #   de forma aleatória e uniforme entre os clientes — ou seja, IID (Independent and Identically Distributed).
    y_off = y_dict["offload_decision"]

    sorted_idx = np.argsort(y_off)
    x = x[sorted_idx]
    y_off = y_off[sorted_idx]
    y_res = y_dict["resource_allocation"][sorted_idx]

    # Divide o dataset igualmente entre os clientes
    data_per_client = len(x) // num_clients
    clients = []

    for i in range(num_clients):
        start = i * data_per_client
        end = (i + 1) * data_per_client
        clients.append((
            x[start:end],
            {
                "offload_decision": y_off[start:end],
                "resource_allocation": y_res[start:end]
            }
        ))

    return clients


# =============================================================================
# PIPELINE COMPLETO PARA FEDERATED LEARNING
# =============================================================================
def prepare_federated_data(config):
    """
    Pipeline completo para preparar dados federados.

    1) Carrega dataset
    2) Cria labels multitarefa
    3) Divide entre clientes
    4) Converte para tf.data.Dataset
    """
    global client_train_datasets
    global client_val_datasets
    global test_dataset

    dataset_name = config["dataset"]["name"]
    num_clients = config["federated_learning"]["total_clients"]
    split_type = config["dataset"]["split"]
    n_actions = config["model"]["heads"]["resource_allocation"]["n_actions"]

    # -------------------------------------------------------------------------
    # Carrega dataset
    # -------------------------------------------------------------------------
    (x_train, y_train), (x_test, y_test) = load_dataset(dataset_name)

    # -------------------------------------------------------------------------
    # Cria labels multitarefa (Multitask labels)
    # -------------------------------------------------------------------------
    y_train_multi = create_multitask_labels(y_train, n_actions)
    y_test_multi = create_multitask_labels(y_test, n_actions)

    # -------------------------------------------------------------------------
    # Split federado
    # -------------------------------------------------------------------------
    if split_type == "iid":
        clients = split_iid(x_train, y_train_multi, num_clients)
    else:
        clients = split_noniid(x_train, y_train_multi, num_clients)

    # -------------------------------------------------------------------------
    # Train / Validation split por cliente
    # -------------------------------------------------------------------------
    client_train_datasets = []
    client_val_datasets = []

    for x_client, y_client in clients:

        split = int(len(x_client) * 0.8)

        train_data = (
            x_client[:split],
            {
                "offload_decision": y_client["offload_decision"][:split],
                "resource_allocation": y_client["resource_allocation"][:split],
            }
        )

        val_data = (
            x_client[split:],
            {
                "offload_decision": y_client["offload_decision"][split:],
                "resource_allocation": y_client["resource_allocation"][split:],
            }
        )

        client_train_datasets.append(train_data)
        client_val_datasets.append(val_data)

    # -------------------------------------------------------------------------
    # Dataset de teste GLOBAL (para avaliação científica)
    # -------------------------------------------------------------------------
    test_dataset = (x_test, y_test_multi)
    print("✔ Federated datasets preparados:", len(client_train_datasets))
    return client_train_datasets, client_val_datasets


# =============================================================================
# ACESSO DOS CLIENTES / CARREGAMENTO POR CLIENTE (FlowerClient) - ADAPTAÇÃO RAY-SAFE
# =============================================================================
def load_client_data(cid, config):

    global client_train_datasets
    global client_val_datasets

    if client_train_datasets is None:

        raise RuntimeError(
            "Datasets não inicializados. Chame prepare_federated_data() antes."
        )

    # número de clientes definido no config
    num_clients = config["federated_learning"]["total_clients"]

    # garante consistência com Flower/Ray
    cid = int(cid) % num_clients

    # proteção extra caso Ray gere mais clientes que datasets
    cid = cid % len(client_train_datasets)

    train_dataset = client_train_datasets[cid]
    val_dataset = client_val_datasets[cid]

    return train_dataset, val_dataset