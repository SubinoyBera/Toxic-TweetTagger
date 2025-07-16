# Shell script to set up a Python virtual environment with specific requirements
# Usage: ./init_setup.sh

echo [$(date)]: "START"

echo [$(date)]: "creating env with python version -- 3.11.5"

# Create a virtual environment- .venv using conda
conda create --prefix ./venv python=3.11.5 -y

echo [$(date)]: "activating virtual environment"

# Activate the virtual environment
source activate ./venv

echo [$(date)]: "installing development requirements"

# Install the development requirements
pip install --upgrade pip
pip install uv
uv pip install -r requirements-dev.txt

echo [$(date)]: "END"