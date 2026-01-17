#!/bin/bash

# CartPole Fuzzy Controller - Setup Script
# This script creates a virtual environment and installs dependencies

echo "=============================================="
echo "CartPole Fuzzy Controller Setup"
echo "=============================================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "=============================================="
echo "Setup complete!"
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the simulation:"
echo "  python main.py [num_episodes]"
echo "=============================================="
