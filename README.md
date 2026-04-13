# CARLA Speed Planning: Rule-based, ANN, SNN Comparison

A comparative study of three speed planning strategies for autonomous driving in the [CARLA simulator](https://carla.org/) (v0.9.16):

- **Rule-based** -- Curvature-inverse analytical formula
- **ANN** -- Multi-Layer Perceptron trained via imitation learning
- **SNN** -- Spiking Neural Network (LIF neurons) trained via imitation learning

Both neural network planners learn from the Rule-based planner using synthetic data, then all three are evaluated on identical CARLA routes across multiple spawn points.

## Project Structure

```
.
├── main.py                  # Entry point: training, evaluation, plotting
├── config.py                # All hyperparameters and configuration
├── requirements.txt
├── planners/
│   ├── rule_planner.py      # Rule-based: K / (1 + α|κ|)
│   ├── ann_planner.py       # ANN: 3→64→64→1 MLP (ReLU)
│   └── snn_planner.py       # SNN: LIF neurons, direct current injection
├── controller/
│   └── pid_controller.py    # PID speed→throttle/brake controller
├── env/
│   └── carla_env.py         # CARLA environment wrapper
├── training/
│   ├── imitation.py         # Imitation learning (synthetic + CARLA data)
│   └── rl_train.py          # RL fine-tuning (experimental)
├── evaluation/
│   └── metrics.py           # Energy, jerk, speed, distance metrics
├── visualization/
│   ├── plot.py              # Matplotlib comparison plots
│   └── debug_draw.py        # CARLA in-world debug HUD overlay
├── utils/
│   └── logger.py            # Episode data logger (CSV + NPZ)
├── models/                  # Saved model weights (.pth)
├── logs/                    # Episode logs (.csv + .npz)
└── plots/                   # Generated comparison plots (.png)
```

## Architecture

### Planners

All three planners take the same 3D state vector `[speed, curvature, distance_to_waypoint]` and output a target speed (m/s).

| Planner | Method | Parameters |
|---------|--------|------------|
| Rule-based | `K / (1 + α \|κ\|)`, clamped to [3, 12] m/s | K=12, α=50 |
| ANN | MLP: Linear(3,64)→ReLU→Linear(64,64)→ReLU→Linear(64,1) | ~4.5K params |
| SNN | BatchNorm→Linear→LIF→Linear→LIF→Linear→LIF, 25 timesteps | ~4.5K params, β=0.85 |

### SNN Details

- **Neuron model**: Leaky Integrate-and-Fire (LIF), membrane decay β=0.85
- **Encoding**: Direct current injection (continuous input repeated every timestep)
- **Readout**: Mean membrane potential of the output layer across 25 timesteps
- **Surrogate gradient**: Fast sigmoid (slope=25) for backpropagation through spikes

### Training

Both ANN and SNN are trained via imitation learning on 20,000 synthetic samples generated from the Rule-based planner:

- **Loss**: MSE
- **Optimizer**: Adam (ANN: lr=1e-3, SNN: lr=5e-4)
- **Scheduler**: CosineAnnealingLR
- **Epochs**: 2000
- **Gradient clipping**: max_norm=1.0

### Control Pipeline

```
State → Planner → Target Speed → PID Controller → Throttle/Brake → CARLA Vehicle
```

PID gains: Kp=1.0, Ki=0.1, Kd=0.05

## Prerequisites

- **CARLA 0.9.16** simulator installed and running
- **Python 3.8+**
- CARLA Python API (`carla` package from the CARLA distribution)

## Installation

```bash
git clone https://github.com/WeiyeTao/carla-speed-planning.git
cd carla-speed-planning
pip install -r requirements.txt
```

## Usage

### Start CARLA server first

```bash
# Linux
./CarlaUE4.sh

# Windows
CarlaUE4.exe
```

### Run the full pipeline (train + evaluate + plot)

```bash
python3 main.py --mode all
```

### Individual stages

```bash
# Train ANN and SNN only (no CARLA needed with synthetic data)
python3 main.py --mode train --synthetic

# Evaluate all planners in CARLA
python3 main.py --mode evaluate

# Generate plots from existing logs
python3 main.py --mode plot
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--mode` | `all`, `train`, `evaluate`, or `plot` | `all` |
| `--host` | CARLA server host | `localhost` |
| `--port` | CARLA server port | `2000` |
| `--episodes` | Episodes per planner per spawn point | `1` |
| `--max-steps` | Max simulation steps per episode | `2000` |
| `--train-epochs` | Training epochs | `2000` |
| `--synthetic` | Use synthetic training data | off |
| `--realtime` | Slow simulation to real-time for visual observation | off |
| `--no-debug` | Disable CARLA debug HUD overlay | off |

## Configuration

All parameters are centralized in `config.py`:

- Pipeline stage toggles and active planners
- CARLA connection settings
- Simulation parameters (dt=0.05s, 20Hz)
- Route settings (500m, 4 spawn points)
- PID gains, energy model coefficients
- Planner architectures and training hyperparameters
- Visualization options

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Total Energy | Cumulative `throttle² + 0.1 × brake` |
| Energy per Meter | Energy normalized by distance traveled |
| Distance Traveled | Accumulated displacement (m) |
| Average Speed | Mean speed over the episode (m/s) |
| Travel Time | Episode duration (s) |
| Mean Absolute Jerk | Smoothness: mean \|d²v/dt²\| |
| Max Absolute Jerk | Peak jerk value |

## Generated Plots

The pipeline produces 7 comparison plots in `plots/`:

- `speed_vs_time.png` -- Speed profiles of all planners
- `energy_vs_time.png` -- Cumulative energy consumption
- `control_signals.png` -- Throttle and brake signals
- `jerk_vs_time.png` -- Jerk (ride comfort) comparison
- `energy_comparison.png` -- Bar chart of total energy and energy/meter
- `metrics_comparison.png` -- Multi-metric bar chart comparison
- `spike_raster.png` -- SNN spike activity visualization

## License

This project is for academic use (ME5423 coursework).
