#!/bin/bash

# Adaptive Scene Setup Script
# This script automates the installation of all dependencies

set -e  # Exit on any error

# Save the project directory at the start
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 Setting up Adaptive Scene project..."

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "⚠️  This script is designed for Linux. Please manual setup on other platforms."
    exit 1
fi

# Determine environment mode
ENV_MODE="local"
if [[ "$1" == "--cluster" ]] || [[ -n "$CLUSTER" ]] || [[ -n "$SLURM_JOB_ID" ]]; then
    ENV_MODE="cluster"
    echo "🖥️  Detected cluster environment"
elif [[ "$1" == "--real" ]]; then
    ENV_MODE="real"
    echo "🤖 Setting up for real robot environment"
else
    echo "💻 Setting up for local Linux environment"
fi

# For backward compatibility
CLUSTER_MODE=false
if [[ "$ENV_MODE" == "cluster" ]]; then
    CLUSTER_MODE=true
fi

# Function to add lines to bashrc if they don't exist
add_to_bashrc() {
    local line="$1"
    if ! grep -Fxq "$line" ~/.bashrc; then
        echo "$line" >> ~/.bashrc
        echo "✅ Added to ~/.bashrc: $line"
    else
        echo "ℹ️  Already in ~/.bashrc: $line"
    fi
}

echo "📝 Configuring environment variables..."

# Add common environment variables to ~/.bashrc
add_to_bashrc ""
add_to_bashrc "# Adaptive Scene Environment Variables"
add_to_bashrc "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\${HOME}/.mujoco/mujoco210/bin"
add_to_bashrc "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/lib/nvidia"
add_to_bashrc "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
add_to_bashrc "export MUJOCO_GL=egl"
add_to_bashrc "export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH"

# Add CoppeliaSim environment variables only for non-cluster environments
if [ "$CLUSTER_MODE" = false ]; then
    add_to_bashrc "export COPPELIASIM_ROOT=\${HOME}/.coppeliasim"
    add_to_bashrc "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$COPPELIASIM_ROOT"
    add_to_bashrc "export QT_QPA_PLATFORM_PLUGIN_PATH=\$COPPELIASIM_ROOT"
fi

# Source the updated bashrc
source ~/.bashrc

echo "📥 Installing MuJoCo..."
mkdir -p ~/.mujoco
cd ~/.mujoco

if [ ! -f "mujoco210.tar.gz" ]; then
    wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco210.tar.gz --no-check-certificate
    echo "✅ Downloaded MuJoCo"
else
    echo "ℹ️  MuJoCo already downloaded"
fi

if [ ! -d "mujoco210" ]; then
    tar -xvzf mujoco210.tar.gz
    echo "✅ Extracted MuJoCo"
else
    echo "ℹ️  MuJoCo already extracted"
fi

# Install CoppeliaSim only for non-cluster environments
if [ "$CLUSTER_MODE" = false ]; then
    echo "📥 Installing CoppeliaSim..."
    cd ~
    if [ ! -f "CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz" ]; then
        wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
        echo "✅ Downloaded CoppeliaSim"
    else
        echo "ℹ️  CoppeliaSim already downloaded"
    fi

    if [ ! -d "$HOME/.coppeliasim" ]; then
        mkdir -p $HOME/.coppeliasim
        tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -C $HOME/.coppeliasim --strip-components 1
        echo "✅ Extracted CoppeliaSim"
    else
        echo "ℹ️  CoppeliaSim already extracted"
    fi

    # Clean up
    rm -f CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
else
    echo "⏭️  Skipping CoppeliaSim installation for cluster environment"
fi

echo "🐍 Setting up Conda environment..."
cd "$PROJECT_DIR"  # Go back to project directory

# Check if mamba is available
if ! command -v mamba &> /dev/null; then
    echo "📦 Installing mamba..."
    conda install -n base -c conda-forge mamba=1.5.9 -y
fi

# Select environment file based on mode
if [ "$ENV_MODE" = "cluster" ]; then
    ENV_FILE="conda_environment_cluster.yaml"
    echo "🖥️  Using cluster environment file"
elif [ "$ENV_MODE" = "real" ]; then
    ENV_FILE="conda_environment_real.yaml"
    echo "🤖 Using real robot environment file"
else
    ENV_FILE="conda_environment.yaml"
    echo "💻 Using local environment file"
fi

# Create environment if it doesn't exist
if ! conda env list | grep -q "policy-consensus"; then
    echo "🔨 Creating policy-consensus environment..."
    mamba env create -f $ENV_FILE
    echo "✅ Created policy-consensus environment"
else
    echo "ℹ️  policy-consensus environment already exists"
fi

echo "📦 Installing package in development mode..."
eval "$(conda shell.bash hook)"
conda activate policy-consensus
cd "$PROJECT_DIR"  # Ensure we're in project directory
pip install -e .

echo ""
echo "🎉 Setup complete! 🎉"
echo ""
echo "To get started:"
echo "1. Restart your terminal or run: source ~/.bashrc"
echo "2. Activate the environment: conda activate policy-consensus"
echo "3. See README.md for usage instructions"
echo ""