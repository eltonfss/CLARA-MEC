# """
# ===============================================================================
# mec_generator.py

# Gerador sintético MEC do MTDA.

# Implementa:

# Source Domain
# Target Domain
# Domain Shift

# Estado MEC:

# s_t^k =
# (B_t^k, C_t^k, H_t^k, P_t^k, D_t^k)

# ===============================================================================
# """

# import numpy as np


# class MECGenerator:

#     def __init__(self, config):

#         self.config = config

#         mec_cfg = config["mec"]

#         self.num_samples = mec_cfg["samples"]

#         self.bandwidth_min = mec_cfg["bandwidth_min"]
#         self.bandwidth_max = mec_cfg["bandwidth_max"]

#         self.cpu_min = mec_cfg["cpu_min"]
#         self.cpu_max = mec_cfg["cpu_max"]

#         self.channel_gain_min = mec_cfg["channel_gain_min"]
#         self.channel_gain_max = mec_cfg["channel_gain_max"]

#         self.power_min = mec_cfg["power_min"]
#         self.power_max = mec_cfg["power_max"]

#         self.task_size_min = mec_cfg["task_size_min"]
#         self.task_size_max = mec_cfg["task_size_max"]

#     # =========================================================================
#     # Source Domain
#     # =========================================================================
#     def generate_source_domain(self):

#         print("📦 Gerando Source Domain...")

#         X = self._generate_base_distribution()

#         return X

#     # =========================================================================
#     # Target Domain (com domain shift)
#     # =========================================================================
#     def generate_target_domain(self):

#         print("🌍 Gerando Target Domain...")

#         X = self._generate_base_distribution()

#         shift_cfg = self.config["domain_shift"]

#         if shift_cfg["enabled"]:

#             X = self.apply_domain_shift(
#                 X,
#                 shift_cfg
#             )

#         return X

#     # =========================================================================
#     # Distribuição MEC Base
#     # =========================================================================
#     def _generate_base_distribution(self):

#         bandwidth = np.random.uniform(
#             self.bandwidth_min,
#             self.bandwidth_max,
#             self.num_samples
#         )

#         cpu = np.random.uniform(
#             self.cpu_min,
#             self.cpu_max,
#             self.num_samples
#         )

#         channel_gain = np.random.uniform(
#             self.channel_gain_min,
#             self.channel_gain_max,
#             self.num_samples
#         )

#         power = np.random.uniform(
#             self.power_min,
#             self.power_max,
#             self.num_samples
#         )

#         task_size = np.random.uniform(
#             self.task_size_min,
#             self.task_size_max,
#             self.num_samples
#         )

#         # =============================================================
#         # Estado MEC
#         # =============================================================

#         X = np.column_stack([
#             bandwidth,
#             cpu,
#             channel_gain,
#             power,
#             task_size
#         ])

#         return X.astype(np.float32)

#     # =========================================================================
#     # Domain Shift
#     # =========================================================================
#     def apply_domain_shift(
#         self,
#         X,
#         shift_cfg
#     ):

#         X_shifted = X.copy()

#         bandwidth_shift = shift_cfg["bandwidth_shift"]
#         cpu_shift = shift_cfg["cpu_shift"]
#         workload_shift = shift_cfg["workload_shift"]
#         noise_level = shift_cfg["noise_level"]

#         # =============================================================
#         # Shift no bandwidth
#         # =============================================================
#         X_shifted[:, 0] *= (
#             1 + bandwidth_shift
#         )

#         # =============================================================
#         # Shift CPU
#         # =============================================================
#         X_shifted[:, 1] *= (
#             1 - cpu_shift
#         )

#         # =============================================================
#         # Shift workload (task size)
#         # =============================================================
#         X_shifted[:, 4] *= (
#             1 + workload_shift
#         )

#         # =============================================================
#         # Noise no canal
#         # =============================================================
#         noise = np.random.normal(
#             0,
#             noise_level,
#             X_shifted.shape
#         )

#         X_shifted += noise

#         return X_shifted.astype(np.float32)

"""
===============================================================================
mec_generator.py

Gerador sintético MEC do MTDA
(versão científica corrigida)

- Source Domain
- Target Domain
- Domain Shift real
- Features MEC não-lineares
- Noise
- Feature Engineering

===============================================================================
"""

import numpy as np

from sklearn.preprocessing import (
    StandardScaler
)


class MECGenerator:

    def __init__(self, config):

        self.config = config

        mec_cfg = config["mec"]

        self.num_samples = mec_cfg["samples"]

        self.bandwidth_min = mec_cfg["bandwidth_min"]
        self.bandwidth_max = mec_cfg["bandwidth_max"]

        self.cpu_min = mec_cfg["cpu_min"]
        self.cpu_max = mec_cfg["cpu_max"]

        self.channel_gain_min = (
            mec_cfg["channel_gain_min"]
        )

        self.channel_gain_max = (
            mec_cfg["channel_gain_max"]
        )

        self.power_min = mec_cfg["power_min"]
        self.power_max = mec_cfg["power_max"]

        self.task_size_min = (
            mec_cfg["task_size_min"]
        )

        self.task_size_max = (
            mec_cfg["task_size_max"]
        )

        self.scaler = StandardScaler()

    # =========================================================================
    # Source Domain
    # =========================================================================
    def generate_source_domain(self):

        print("📦 Gerando Source Domain...")

        X = self._generate_base_distribution()

        X = self.scaler.fit_transform(X)

        return X.astype(np.float32)

    # =========================================================================
    # Target Domain
    # =========================================================================
    def generate_target_domain(self):

        print("🌍 Gerando Target Domain...")

        X = self._generate_base_distribution()

        shift_cfg = self.config[
            "domain_shift"
        ]

        if shift_cfg["enabled"]:

            X = self.apply_domain_shift(
                X,
                shift_cfg
            )

        X = self.scaler.transform(X)

        return X.astype(np.float32)

    # =========================================================================
    # MEC Base Distribution
    # =========================================================================
    def _generate_base_distribution(self):

        bandwidth = np.random.uniform(
            self.bandwidth_min,
            self.bandwidth_max,
            self.num_samples
        )

        cpu = np.random.uniform(
            self.cpu_min,
            self.cpu_max,
            self.num_samples
        )

        channel_gain = np.random.uniform(
            self.channel_gain_min,
            self.channel_gain_max,
            self.num_samples
        )

        power = np.random.uniform(
            self.power_min,
            self.power_max,
            self.num_samples
        )

        task_size = np.random.uniform(
            self.task_size_min,
            self.task_size_max,
            self.num_samples
        )

        # ==========================================================
        # Feature engineering MEC
        # (deixa o problema não trivial)
        # ==========================================================

        transmission_cost = (
            task_size /
            (bandwidth + 1e-6)
        )

        cpu_ratio = (
            cpu /
            (task_size + 1e-6)
        )

        log_task = np.log1p(
            task_size
        )

        noise = np.random.normal(
            0,
            0.15,
            self.num_samples
        )

        # ==========================================================
        # Estado MEC enriquecido
        # ==========================================================

        X = np.column_stack([

            bandwidth,

            cpu,

            channel_gain,

            power,

            task_size,

            transmission_cost,

            cpu_ratio,

            log_task,

            noise
        ])

        return X.astype(np.float32)

    # =========================================================================
    # Domain Shift
    # =========================================================================
    def apply_domain_shift(
        self,
        X,
        shift_cfg
    ):

        X_shifted = X.copy()

        # bandwidth
        X_shifted[:, 0] *= 0.55

        # cpu
        X_shifted[:, 1] *= 0.65

        # channel gain
        X_shifted[:, 2] *= 0.60

        # power
        X_shifted[:, 3] *= 1.35

        # workload
        X_shifted[:, 4] *= 2.2

        # ==========================================================
        # Noise pesado no target
        # ==========================================================

        noise = np.random.normal(
            0,
            0.25,
            X_shifted.shape
        )

        X_shifted += noise

        return X_shifted.astype(np.float32)