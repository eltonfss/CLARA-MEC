"""
===============================================================================
config_loader.py

Carrega configuração YAML do MTDA.
===============================================================================
"""

import yaml


def load_config(config_path="config.yaml"):

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config