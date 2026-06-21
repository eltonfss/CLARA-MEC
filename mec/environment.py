"""
===============================================================================
Arquivo: environment.py

Descrição:
Ambiente formal do CLARA-MEC modelado como um MDP consistente com a
Section III do artigo submetido à Wiley.

Implementa:

- Estado S_t = {D_t, C_local,t, E_t, B_t}
- Ação multitarefa a_t = (d_t, r_t)
- Modelo formal de latência
- Modelo formal de energia
- Função de recompensa explícita
- Transição de estado coerente

Autor: Nelson Machado Junior
===============================================================================
"""

import numpy as np

class MECEnvironment:

    def __init__(self, config):

        self.config = config

        # -------------------------------
        # Parâmetros físicos
        # -------------------------------
        self.kappa = float(config["env"]["kappa"])            # constante DVFS
        self.tx_power = float(config["env"]["tx_power"])      # potência de transmissão
        self.alpha = float(config["env"]["alpha"])            # peso latência
        self.beta = float(config["env"]["beta"])              # peso energia
        # self.edge_cpu_levels = config["env"]["edge_cpu_levels"]
        self.edge_cpu_levels = [float(x) for x in config["env"]["edge_cpu_levels"]]
        self.snr = float(config["env"]["snr"])

        # -------------------------------
        # Parâmetros de tarefa
        # -------------------------------
        self.size_min = float(config["task"]["size_min"])
        self.size_max = float(config["task"]["size_max"])
        self.cycles_min = float(config["task"]["cycles_min"])
        self.cycles_max = float(config["task"]["cycles_max"])

        # -------------------------------
        # Estado inicial
        # -------------------------------
        self.reset()

    # ---------------------------------------------------------
    # Geração de tarefa estocástica
    # ---------------------------------------------------------
    def generate_task(self):

        task_size = np.random.uniform(
            self.size_min,
            self.size_max
        )

        task_complexity = np.random.uniform(
            self.cycles_min,
            self.cycles_max
        )

        return task_size, task_complexity


    # =========================================================================
    # RESET DO AMBIENTE
    # =========================================================================
    def reset(self):
        """
        Inicializa um novo estado do sistema.
        """        

        self.task_size, self.task_complexity = self.generate_task()

        self.local_cpu = np.random.uniform(1e9, 3e9)
        self.bandwidth = np.random.uniform(5e6, 20e6)
        self.energy = np.random.uniform(50, 100)

        return self._get_state()


    # =========================================================================
    # ESTADO
    # =========================================================================
    def _get_state(self):

        return np.array([
            self.task_size,
            self.task_complexity,
            self.local_cpu,
            self.bandwidth,
            self.energy
        ])


    # =========================================================================
    # STEP
    # =========================================================================
    def step(self, action):
        """
        Executa ação multitarefa:

        action = (offload_decision, resource_level)
        """        

        offload_decision, resource_level = action

        # -----------------------------------
        # Taxa de transmissão (Shannon)
        # -----------------------------------
        # Isso aqui ignora o channel fading, ou seja, SNR constante, o que gera latência de rede quase 
        # constante, o que não é realista.
        # transmission_rate = self.bandwidth * np.log2(1 + self.snr) 
        
        # Isso aqui cria channel variability, que é padrão em papers MEC.
        # É apenas melhoria de realismo, não erro grave.
        snr = np.random.exponential(self.snr)  
        transmission_rate = self.bandwidth * np.log2(1 + snr)

        # -------------------------
        # Execução local
        # -------------------------
        cycles = self.task_size * self.task_complexity

        if offload_decision == 0:
            latency = cycles / self.local_cpu

            energy_consumed = (
                self.kappa
                * (self.local_cpu ** 2)
                * cycles
            )

        # -------------------------
        # Offloading
        # -------------------------
        else:

            edge_cpu = self.edge_cpu_levels[resource_level]

            transmission_time = self.task_size / transmission_rate

            edge_time = (
                self.task_size * self.task_complexity
            ) / edge_cpu

            latency = transmission_time + edge_time

            energy_consumed = self.tx_power * transmission_time

        # -----------------------------------
        # Custo e recompensa
        # -----------------------------------
        cost = self.alpha * latency + self.beta * energy_consumed
        reward = -cost

        # -------------------------------------------------------------
        # Atualização do estado (transição) / atualização de energia
        # -------------------------------------------------------------
        self.energy = max(0, self.energy - energy_consumed)

        # -------------------------
        # Nova tarefa
        # -------------------------
        self.task_size, self.task_complexity = self.generate_task()

        done = self.energy <= 0

        next_state = self._get_state()

        info = {
            "latency": latency,
            "energy_consumed": energy_consumed,
            "cost": cost
        }

        return next_state, reward, done, info