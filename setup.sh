#!/usr/bin/env bash

# -----------------------------------------------------------------------------
# Project Name: Neural Splines
# File: setup.sh
# Author: Robert Sitton
# Contact: rsitton@quholo.com
#
# This file is part of a project licensed under the GNU Affero General Public License.
#
# GNU AFFERO GENERAL PUBLIC LICENSE
# Version 3, 19 November 2007
#
# Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
# Everyone is permitted to copy and distribute verbatim copies
# of this license document, but changing it is not allowed.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# setup.sh
#
# This shell script prepares a Python environment for running the Neural
# Splines project on Ubuntu 24.04.  It installs system dependencies,
# creates a dedicated virtual environment, and installs the required
# Python packages including PyTorch and torchvision.  It is
# intended to be idempotent and safe to run multiple times.
#
# Usage:
#     bash setup.sh
#
# The script must be executed from the root of the repository.  Feel
# free to edit the versions below to target different releases of
# PyTorch.  Always consult the official installation documentation
# for the most accurate instructions.
# -----------------------------------------------------------------------------

set -euo pipefail

echo "[NeuralSplines] Updating package lists..."
sudo apt-get update -y

echo "[NeuralSplines] Installing system packages..."
sudo apt-get install -y \
    python3 \
    python3-venv \
    python3-pip \
    build-essential \
    git

# Create a Python virtual environment in the .venv directory.  If the
# directory already exists the command is skipped.  Users may choose
# another location by modifying the VENV_DIR variable.
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[NeuralSplines] Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

echo "[NeuralSplines] Activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo "[NeuralSplines] Upgrading pip and setuptools..."
python -m pip install --upgrade pip setuptools wheel

# Install Python packages.  You can pin specific versions here for
# reproducibility.  Torch and torchvision with CPU support are
# installed via the PyTorch wheels index.
TORCH_VERSION="2.3.1"
TORCHVISION_VERSION="0.18.1"

echo "[NeuralSplines] Installing Python dependencies..."
python -m pip install \
    torch==${TORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} \
    matplotlib \
    tqdm

echo "[NeuralSplines] Setup complete.  To activate the environment later run"
echo "    source $VENV_DIR/bin/activate"