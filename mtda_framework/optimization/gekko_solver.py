"""
===============================================================================
gekko_solver.py

Solver MINLP do MTDA via GEKKO.

Resolve:

1. decisão ótima de offloading
2. alocação ótima de recurso
3. custo ótimo MEC

Conforme artigo:

C_t^k =
αL_t^k + βE_t^k

===============================================================================
"""

import numpy as np
from gekko import GEKKO


class GekkkoMECOptimizer:

    def __init__(self, config):

        self.config = config

        self.alpha = (
            config["cost_function"]["alpha_latency"]
        )

        self.beta = (
            config["cost_function"]["beta_energy"]
        )

    # =========================================================================
    # Solve MINLP
    # =========================================================================
    def solve_dataset(self, X):

        print("⚙️ Resolvendo MINLP via GEKKO...")

        offloading_labels = []
        resource_labels = []

        latency_values = []
        energy_values = []
        cost_values = []

        total = len(X)

        for idx, sample in enumerate(X):

            if idx % 1000 == 0:
                print(
                    f"[{idx}/{total}]"
                )

            result = self.solve_sample(sample)

            offloading_labels.append(
                result["offload_decision"]
            )

            resource_labels.append(
                result["resource_allocation"]
            )

            latency_values.append(
                result["latency"]
            )

            energy_values.append(
                result["energy"]
            )

            cost_values.append(
                result["cost"]
            )

        labels = {
            "offload_decision":
                np.array(offloading_labels),

            "resource_allocation":
                np.array(resource_labels),

            "latency":
                np.array(latency_values),

            "energy":
                np.array(energy_values),

            "cost":
                np.array(cost_values)
        }

        print("✔ Labels ótimos gerados.\n")

        return labels

    # =========================================================================
    # Solve single MEC state
    # =========================================================================
    def solve_sample(self, sample):

        bandwidth = float(sample[0])
        cpu = float(sample[1])
        channel_gain = float(sample[2])
        power = float(sample[3])
        task_size = float(sample[4])

        m = GEKKO(remote=False)

        # =============================================================
        # Variáveis de decisão
        # =============================================================

        # Offloading decision
        x = m.Var(
            value=1,
            lb=0,
            ub=1,
            integer=True
        )

        # Resource allocation
        r = m.Var(
            value=0.5,
            lb=0.1,
            ub=1.0
        )

        # =============================================================
        # MEC equations
        # =============================================================

        # Transmission rate
        transmission_rate = (
            bandwidth *
            np.log2(
                1 +
                power * channel_gain
            )
        )

        transmission_rate = max(
            transmission_rate,
            1e-6
        )

        # =============================================================
        # Latency
        # =============================================================

        local_latency = (
            task_size / cpu
        )

        offload_latency = (
            task_size /
            transmission_rate
        ) + (
            task_size /
            (cpu * r)
        )

        latency = (
            (1 - x)
            * local_latency
            +
            x
            * offload_latency
        )

        # =============================================================
        # Energy
        # =============================================================

        local_energy = (
            power *
            local_latency
        )

        offload_energy = (
            power *
            (
                task_size /
                transmission_rate
            )
        )

        energy = (
            (1 - x)
            * local_energy
            +
            x
            * offload_energy
        )

        # =============================================================
        # Cost function
        # =============================================================

        cost = (
            self.alpha
            * latency
            +
            self.beta
            * energy
        )

        # =============================================================
        # Objective
        # =============================================================

        m.Minimize(cost)

        # =============================================================
        # Solver
        # =============================================================

        m.options.SOLVER = 1
        m.options.MAX_ITER = 100

        try:

            m.solve(disp=False)

            decision = int(round(x.value[0]))

            resource = float(r.value[0])

            latency_value = float(latency.value[0])
            energy_value = float(energy.value[0])
            cost_value = float(cost.value[0])

        except Exception:

            decision = 0
            resource = 0.5

            latency_value = 0.0
            energy_value = 0.0
            cost_value = 0.0

        return {
            "offload_decision":
                decision,

            "resource_allocation":
                resource,

            "latency":
                latency_value,

            "energy":
                energy_value,

            "cost":
                cost_value
        }