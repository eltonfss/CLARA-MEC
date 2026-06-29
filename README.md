# CLARA-MEC

CLARA-MEC is an experimental Python framework for evaluating **continual federated learning**, **multi-task decision policies**, and **mobile edge computing (MEC) offloading**. The project combines Flower-based federated learning simulations, TensorFlow/Keras policy models, MEC latency/energy/cost modeling, and baseline strategies for comparative experiments.

The codebase is organized around a configurable experiment pipeline: `config.yaml` defines the dataset, model, federated setup, MEC environment, and continual-learning coefficients; `train.py` loads that configuration, prepares federated client data, starts a Flower simulation, and writes round-level metrics to `results/`.

## Key Features

- **Federated learning with Flower**: supports FedAvg, FedSGD, FedProx, personalized FL-style partial parameter sharing, a heuristic offloading baseline, and the CLARA-MEC strategy.
- **Continual learning regularization**: CLARA-MEC augments supervised multi-task training with Knowledge Distillation (KD) and Elastic Weight Consolidation (EWC).
- **Multi-task policy model**: a TensorFlow/Keras CNN backbone predicts both binary offloading decisions and discrete resource-allocation actions.
- **MEC environment simulation**: models stochastic task generation, local execution, edge offloading, latency, energy consumption, cost, reward, and energy depletion.
- **Dataset support**: uses Keras datasets for MNIST, Fashion-MNIST, CIFAR-10, and CIFAR-100, with IID and non-IID federated client splits.
- **Metrics and plotting**: records loss, accuracy, latency, energy, and cost to CSV, and generates metric plots under `results/plots/`.
- **MTDA-related baseline framework**: includes centralized and standalone MTDA components under `baselines/` and `mtda_framework/`.

## Repository Layout

```text
.
├── config.yaml                  # Main CLARA-MEC experiment configuration
├── train.py                     # Main Flower federated simulation entry point
├── run_experiments.py           # Convenience runner for the configured dataset/strategy
├── plot_results.py              # Generates plots from the latest metrics CSV
├── requirements.txt             # Runtime dependency list
├── pyproject.toml               # Package metadata and dependencies
├── baselines/                   # Centralized MTDA and heuristic offloading baselines
├── federated/                   # Flower client, FL strategies, KD/EWC continual learning
├── mec/                         # MEC MDP environment
├── models/                      # Multi-task policy model
├── utils/                       # Data loading, seeding, and metrics logging
├── mtda_framework/              # Standalone MTDA experiment framework
└── results/                     # Example/generated metrics and plots
```

## Architecture Overview

### 1. Configuration-driven experiments

The main experiment settings live in `config.yaml`:

- `experiment`: experiment name, strategy, random seed, and number of FL rounds.
- `dataset`: dataset name, optional sample count, and IID/non-IID split mode.
- `federated_learning`: total clients, client fraction, local epochs, batch size, and aggregation mode.
- `model`: input shape, CNN backbone parameters, and output-head definitions.
- `continual_learning`: KD temperature and EWC/KD weights.
- `env`, `task`, and `cost_function`: MEC physical/task parameters and cost weights.
- `logging`: controls metric/model persistence flags.

By default, the checked-in configuration runs the `claramec` strategy on `CIFAR10` with 50 clients, 30 federated rounds, and non-IID splitting.

### 2. Data preparation

`utils/data_loader.py` loads a Keras vision dataset, normalizes inputs, creates synthetic multi-task labels, and partitions training data across clients:

- `offload_decision`: binary label derived from the original class label parity.
- `resource_allocation`: discrete resource class derived from the original class modulo the configured number of actions.

The federated split can be:

- `iid`: randomized and evenly divided among clients.
- `noniid`: sorted by offloading label before partitioning to simulate heterogeneous clients.

Each client partition is split into local train and validation subsets.

### 3. Multi-task policy model

`models/multitask_policy_model.py` defines a CNN-based TensorFlow/Keras model with:

- shared convolutional feature extraction,
- shared dense layers,
- an `offloading_logits` head for binary offloading,
- a `resource_logits` head for resource allocation.

The output heads intentionally return raw logits so that the same outputs can be used consistently for supervised losses and knowledge distillation.

### 4. Federated client training

`federated/flower_client.py` implements a Flower `NumPyClient`. During each local fit round, the client:

1. receives global parameters,
2. trains on its local dataset,
3. computes supervised offloading and resource-allocation losses,
4. optionally adds KD and EWC terms for the CLARA-MEC strategy,
5. interacts with the MEC environment using predicted or heuristic actions,
6. updates Fisher information for EWC,
7. returns updated parameters to the server.

Evaluation reports loss, offloading accuracy, average latency, average energy consumption, and average cost.

### 5. Federated strategies

`federated/strategies.py` provides strategy factories selected by `experiment.strategy`:

| Strategy value | Description |
| --- | --- |
| `fedavg` | Standard FedAvg-style local training and weighted metric aggregation. |
| `fedsgd` | FedAvg configured with one local epoch and small batch behavior. |
| `fedprox` | Flower FedProx with proximal regularization. |
| `pfl` | Personalized-FL-style behavior where clients can retain personalized head weights. |
| `heuristic` | Federated training with rule-based offloading decisions during MEC interaction. |
| `claramec` | Main strategy: FedAvg-style aggregation plus client-side KD and EWC. |

### 6. MEC environment

`mec/environment.py` models a stochastic MEC setting. At each step, an action `(offload_decision, resource_level)` is evaluated as either local execution or edge offloading. The environment computes:

- transmission rate with stochastic SNR variation,
- local or edge latency,
- local computation or transmission energy,
- weighted cost,
- reward as negative cost,
- next task and energy state.

## Installation

### Prerequisites

- Python 3.10 or newer
- A virtual environment is strongly recommended

# NVIDIA GPU Setup for TensorFlow with UV (Ubuntu)

This guide documents the step-by-step configuration performed to enable hardware acceleration (GPU) for TensorFlow using the `uv` package manager on Ubuntu.

---

## 📋 System Prerequisites

Before integrating with Python, the native graphics drivers were installed and verified directly on the host system:

1. **NVIDIA Driver:** Installed via Ubuntu's package manager (`sudo ubuntu-drivers autoinstall`).
2. **Driver Verification:** Running the following command displays the GPU details (**GeForce RTX 2060**):
   ```bash
   nvidia-smi
   ```

---

## 🛠️ Environment Installation

The virtual environment was built using modern python packages that embed CUDA dependencies directly inside the virtual environment:

```bash
# Install TensorFlow with integrated CUDA support
uv pip install "tensorflow[and-cuda]"
```

---

## 🔗 Symbolic Link Adjustment (VIRTUAL_ENV)

To ensure TensorFlow's internal kernel compiler detects the NVIDIA binary paths, a symbolic link was created for the `ptxas` component:

```bash
ln -sf $(find $(dirname $(dirname $(uv run python -c "import nvidia.cuda_nvcc; print(nvidia.cuda_nvcc.__file__)"))) -name ptxas -print -quit) .venv/bin/ptxas
```

---

## 🚀 Resolving the Initialization Error (`Cannot dlopen some GPU libraries`)

The `uv` manager enforces strict environment isolation. To allow TensorFlow to discover the shared NVIDIA libraries (`.so`) installed within your Python package tree, you must explicitly export the correct routes to `LD_LIBRARY_PATH`.

### Persistent Configuration (Zsh)

To make the environment variable configuration permanent across any new terminal session, add the paths to your shell profile:

1. Open your shell configuration file:
   ```bash
   nano ~/.zshrc
   ```

2. Append the following block at the very end of the file:
   ```bash
   # Map NVIDIA internal libraries for TensorFlow under UV
   export LD_LIBRARY_PATH="$HOME/Code/CLARA-MEC/.venv/lib/python3.11/site-packages/nvidia/cudnn/lib:$HOME/Code/CLARA-MEC/.venv/lib/python3.11/site-packages/nvidia/cublas/lib:$HOME/Code/CLARA-MEC/.venv/lib/python3.11/site-packages/nvidia/cufft/lib:$HOME/Code/CLARA-MEC/.venv/lib/python3.11/site-packages/nvidia/curand/lib:$HOME/Code/CLARA-MEC/.venv/lib/python3.11/site-packages/nvidia/cusolver/lib:$HOME/Code/CLARA-MEC/.venv/lib/python3.11/site-packages/nvidia/cusparse/lib:$HOME/Code/CLARA-MEC/.venv/lib/python3.11/site-packages/nvidia/nccl/lib:$HOME/Code/CLARA-MEC/.venv/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH"
   ```

3. Reload the active terminal session configuration:
   ```bash
   source ~/.zshrc
   ```

---

## 🧪 Validation Script

To test if the library mapping is working and your GPU is properly recognized by the library framework, run the verification script:

```bash
uv run test_gpu.py
```

### Validation Code Utilized (`test_gpu.py`):
```python
import tensorflow as tf

print("TensorFlow Version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print(f"\n✅ Success! TensorFlow is using the GPU: {gpus}")
else:
    print("\n❌ GPU not detected. Please verify environment paths.")
```


### Install dependencies

Using `pip` and `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Alternatively, install from `pyproject.toml` in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

> Note: The first run may download TensorFlow/Keras datasets such as CIFAR-10 or MNIST.

## Quick Start

Run the default experiment from `config.yaml`:

```bash
python train.py
```

Or use the convenience wrapper:

```bash
python run_experiments.py
```

After training, metrics are saved to `results/` with filenames like:

```text
CLARA-MEC_metrics_YYYY-MM-DD_HH-MM-SS_CIFAR10.csv
```

Generate plots from the latest CSV in `results/`:

```bash
python plot_results.py
```

Plots are written to `results/plots/`.

## Common Configuration Changes

### Change the dataset

Edit `config.yaml`:

```yaml
dataset:
  name: MNIST      # MNIST, FMNIST, CIFAR10, or CIFAR100
  split: iid       # iid or noniid
```

For MNIST or Fashion-MNIST, also set the model input shape:

```yaml
model:
  input_shape: [28, 28, 1]
```

For CIFAR-10 or CIFAR-100:

```yaml
model:
  input_shape: [32, 32, 3]
```

`run_experiments.py` includes helper logic to update the input shape when switching datasets through its `update_dataset()` function.

### Change the federated strategy

Edit:

```yaml
experiment:
  strategy: claramec
```

Supported values are `fedavg`, `fedsgd`, `fedprox`, `pfl`, `heuristic`, and `claramec`.

### Reduce runtime for smoke tests

For a faster local check, reduce the number of clients, rounds, and local epochs:

```yaml
experiment:
  rounds: 1

federated_learning:
  total_clients: 2
  client_fraction: 1.0
  local_epochs: 1
  batch_size: 32
```

## Outputs

Training produces CSV metrics with the columns:

- `round`
- `loss`
- `accuracy`
- `latency`
- `energy`
- `cost`

Plot generation produces one PNG per available metric.

## Reproducibility

The project sets random seeds from `experiment.seed` and uses `utils/seed.py` to seed Python, NumPy, and TensorFlow. Each Flower client also offsets the seed by client ID for deterministic per-client initialization behavior.

## Development Notes

- TensorFlow threading is limited in `train.py` to reduce resource pressure during simulation.
- Flower/Ray simulation resources are configured in `train.py` with constrained CPU usage.
- `results/` contains generated and example outputs; repeated runs will add timestamped CSV and PNG files.
- The code comments are primarily in Portuguese and document the intended research rationale for many implementation choices.

## Troubleshooting

### Dataset download or network failures

If TensorFlow cannot download a Keras dataset, verify network access or pre-populate the Keras dataset cache in your environment.

### Shape mismatch when changing datasets

Make sure `model.input_shape` matches the dataset:

- MNIST/FMNISt: `[28, 28, 1]`
- CIFAR-10/CIFAR-100: `[32, 32, 3]`

### Long runtime or high memory usage

Federated simulations with many clients, many rounds, and large local epochs can be expensive. Reduce `total_clients`, `rounds`, `local_epochs`, or use a smaller dataset for iteration.

## License

No explicit license file is currently included in this repository. Add a license before distributing or publishing the project.
