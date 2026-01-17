#!/bin/bash

# CartPole Fuzzy Controller - Run Simulation Script
# This script activates the virtual environment and runs the simulation

echo "=============================================="
echo "CartPole Fuzzy Controller"
echo "=============================================="

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found"
    echo "Please run setup.sh first"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Default values
NUM_EPISODES=${1:-5}
SHOW_PLOTS=${2:-true}

echo ""
echo "Configuration:"
echo "  Number of episodes: $NUM_EPISODES"
echo "  Show plots: $SHOW_PLOTS"
echo ""

# Run the simulation
echo "Starting simulation..."
echo ""

python main.py "$NUM_EPISODES" "$SHOW_PLOTS"

# Deactivate virtual environment
deactivate

echo ""
echo "=============================================="
echo "Simulation finished"
echo "=============================================="
