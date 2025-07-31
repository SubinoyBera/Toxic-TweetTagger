# Shell script to set up a Python virtual environment with specific requirements
# Usage: ./init_setup.sh

set -e

echo [$(date)]: "START"

ENV_PATH="./venv"
PYTHON_VERSION="3.11.5"

echo [$(date)]: "Creating conda env at ${ENV_PATH} with Python ${PYTHON_VERSION}"
conda create --prefix "${ENV_PATH}" python="${PYTHON_VERSION}" -y

echo [$(date)]: "activating virtual environment"

# Activate the virtual environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PATH}"

echo [$(date)]: "installing development requirements"

# Install the development requirements
pip install --upgrade pip
pip install uv
uv pip install -r requirements-dev.txt

echo [$(date)]: "DONE"