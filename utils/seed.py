"""
===============================================================================
Arquivo: seed.py

Descrição:
Controle de reprodutibilidade para experimentos científicos no CLARA-MEC.

Garante que todos os experimentos federados sejam determinísticos quando
uma seed é definida no config.yaml.

Controla:

- Python random
- NumPy
- TensorFlow
- Hash seed do Python
===============================================================================
"""

import os
import random
import numpy as np
import tensorflow as tf


def set_global_seed(seed=None):
    """
    Define seed global para reprodutibilidade.

    Parâmetros
    ----------
    seed : int (opcional)
        valor da seed
    """

    if seed is None:
        print("[Seed] Nenhuma seed definida.")
        return

    seed = int(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Determinismo TensorFlow
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    # Evita explosão de threads no Mac / Ray
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    print(f"[Seed] Global seed definida: {seed}")