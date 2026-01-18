# CartPole Fuzzy Control System

A Mamdani-type Fuzzy Logic Control System for balancing the classic CartPole environment using continuous force control.

## Overview

This project implements a fuzzy inference system to solve the CartPole balancing problem. Instead of using discrete left/right actions, the controller applies continuous force to the cart based on fuzzy logic rules that consider:

- **Pole angle** - How far the pole has tilted from vertical
- **Angular velocity** - How fast the pole is rotating
- **Cart position** - How far the cart has drifted from center
- **Cart velocity** - How fast the cart is moving

The fuzzy controller uses 24 rules to determine the appropriate force to apply, combining angle stabilization with drift correction via integral control.

## Project Structure

```
CartPole-Fuzzy-Control-System/
├── main.py                 # Main entry point and simulation loop
├── fuzzy_controller.py     # Fuzzy logic control system implementation
├── cartpole_env.py         # Custom continuous CartPole environment
├── cartpole_parameters.py  # Configuration and parameter definitions 
├── visualization.py        # Plotting and visualization functions
├── requirements.txt        # Python dependencies
├── setup.sh                # Linux virtual environment setup script
├── run_simulation.sh       # Linux simulation execution script
└── README.md               # This file
```

## Requirements

- Python 3.10 or higher
- NumPy
- scikit-fuzzy
- Matplotlib
- Gymnasium (with classic-control)

## Installation

### Windows (Conda Environment)

1. **Install Miniconda or Anaconda**

   Download and install from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

2. **Open Anaconda Prompt** (or terminal with conda initialized)

3. **Create a new conda environment**
   ```bash
   conda create -n cartpole-fuzzy python=3.11
   ```

4. **Activate the environment**
   ```bash
   conda activate cartpole-fuzzy
   ```

5. **Navigate to the project directory**
   ```bash
   cd path\to\CartPole-Fuzzy-Control-System
   ```

6. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

7. **Run the simulation**
   ```bash
   python main.py
   ```

   Or with specific parameters:
   ```bash
   python main.py 5 true    # 5 episodes, show plots
   python main.py 10 false  # 10 episodes, no plots
   ```

### Linux (Virtual Environment)

#### Option A: Using setup scripts (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd CartPole-Fuzzy-Control-System
   ```

2. **Run the setup script**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. **Run the simulation**
   ```bash
   chmod +x run_simulation.sh
   ./run_simulation.sh
   ```

   Or with parameters:
   ```bash
   ./run_simulation.sh 5 true   # 5 episodes, show plots
   ```

#### Option B: Manual setup

1. **Create virtual environment**
   ```bash
   python3 -m venv venv
   ```

2. **Activate the environment**
   ```bash
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Run the simulation**
   ```bash
   python main.py
   ```

### Linux (Conda Environment)

1. **Install Miniconda**
   ```bash
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh
   ```

2. **Create and activate environment**
   ```bash
   conda create -n cartpole-fuzzy python=3.11
   conda activate cartpole-fuzzy
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the simulation**
   ```bash
   python main.py
   ```

## Usage

### Basic Usage

```bash
python main.py [num_episodes] [show_plots]
```

**Arguments:**
- `num_episodes` (optional): Number of episodes to run (default: runs continuously until Ctrl+C)
- `show_plots` (optional): Whether to display visualization plots - `true` or `false` (default: `true`)

### Examples

```bash
# Run with default settings (continuous mode with plots)
python main.py

# Run 10 episodes with plots
python main.py 10 true

# Run 20 episodes without plots (faster)
python main.py 20 false
```

### Stopping the Simulation

Press `Ctrl+C` to stop the simulation. A summary of all episodes will be displayed.

## How It Works

### Fuzzy Control System

The controller uses a Mamdani fuzzy inference system with:

**Input Variables:**
- Pole angle: [-1, 1] radians with 5 membership functions (NL, NS, Z, PS, PL)
  - Symmetric design with overlap for smooth transitions
  - Active region: approximately ±0.1 radians from vertical
- Angular velocity: [-3, 3] rad/s with 3 membership functions (N, Z, P)
- Cart position: [-3, 3] meters with 3 membership functions (N, Z, P)
- Cart velocity: [-1, 1] m/s with 3 membership functions (N, Z, P)
  - Symmetric design with proper overlap at transition points

**Output Variable:**
- Force: [-10, 10] Newtons with 5 membership functions (NL, NS, Z, PS, PL)

**Control Rules:**
- 18 rules for pole balancing (based on pole angle and angular velocity)
  - Large angles trigger aggressive corrective force
  - Small angles use proportional response based on angular velocity
- 6 rules for cart centering (based on cart position and velocity)
  - Push cart back toward center when it drifts

### Integral Control

An integral controller is included to eliminate steady-state position drift. It accumulates position error when the pole is nearly vertical (below threshold) and applies corrective force to center the cart. Features include:
- Anti-windup limiting to prevent excessive integral buildup
- Automatic decay when pole becomes unstable (prioritizes balancing over centering)

## Visualization

When `show_plots` is enabled, the following visualizations are generated:

1. **Membership Functions** - Shows all 5 fuzzy variables with their membership functions
   - Pole angle plot is zoomed to ±0.15 radians to show detail in the active region
   - All plots include reference lines for clarity
2. **Control Surface** - 3D plot of force output vs angle and angular velocity
   - Shows the fuzzy controller response with cart at center position
3. **Simulation Results** - Time-series plots of angle, velocity, position, and control actions
4. **Episode Summary** - Bar chart comparing episode performance

## Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| numpy | >=1.24.0 | Numerical computations |
| scikit-fuzzy | >=0.4.2 | Fuzzy logic implementation |
| matplotlib | >=3.7.0 | Data visualization |
| gymnasium | >=0.29.0 | CartPole environment |

## Known Issues

### Cart Drifting

The control system currently has a known issue where the cart drifts away from the center position over time. While the integral controller attempts to correct this drift, it does not fully eliminate the problem. The cart may gradually move towards the edge of the track during longer episodes.

**Status:** This issue is under investigation and needs to be fixed in a future update.

## Troubleshooting

### TkAgg Backend Issues (Linux)

If you encounter matplotlib backend errors, install tkinter:
```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# Fedora
sudo dnf install python3-tkinter

# Arch Linux
sudo pacman -S tk
```

### Conda Environment Not Activating (Windows)

Ensure you're using Anaconda Prompt or run:
```bash
conda init powershell
```
Then restart your terminal.

### Display Issues (Headless Linux)

For running on servers without display:
```bash
python main.py 10 false  # Disable plots
```

Or set the matplotlib backend:
```bash
export MPLBACKEND=Agg
python main.py
```

## License

This project is provided for educational purposes.
