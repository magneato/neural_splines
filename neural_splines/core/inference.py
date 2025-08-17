#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Project Name: Neural Splines
# File: inference.py
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

"""
inference.py
============

Evaluate a trained dense neural network on the MNIST test set.  This
script is independent of the spline implementation and assumes that
you have already converted a spline model to its dense equivalent
using :func:`SplineMLP.to_dense_mlp`.  It corresponds to the
"Densification and Inference" discussion in Section 6 of the LaTeX
document.  Refer to that section for a high‑level rationale of
densification and how it removes the need for interpolation during
inference.

Usage example::

    python3 inference.py \
        --model-path checkpoints/dense_model.pth \
        --input-size 784 --hidden-size 128 --output-size 10 \
        --batch-size 256

The script reconstructs a dense multilayer perceptron with the
specified dimensions, loads the saved weights, and computes the
classification accuracy on the MNIST test set.  Adjust ``input-size``,
``hidden-size`` and ``output-size`` to match the architecture used
during training.  If loading the model or downloading the dataset
fails, a descriptive error is raised.
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from neural_spline import DenseMLP


def get_test_loader(batch_size: int) -> DataLoader:
    """Return a DataLoader over the MNIST test set.

    Parameters
    ----------
    batch_size : int
        Number of images per batch.

    Returns
    -------
    DataLoader
        Data loader for the MNIST test set.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    return DataLoader(test_set, batch_size=batch_size, shuffle=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with a dense MLP on MNIST")
    parser.add_argument("--model-path", type=str, required=True, help="path to dense model state dict")
    parser.add_argument("--input-size", type=int, default=784, help="number of input features")
    parser.add_argument("--hidden-size", type=int, default=128, help="number of hidden units")
    parser.add_argument("--output-size", type=int, default=10, help="number of output units")
    parser.add_argument("--batch-size", type=int, default=256, help="batch size for evaluation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Reconstruct the architecture used during training
    layer1 = nn.Linear(args.input_size, args.hidden_size)
    layer2 = nn.Linear(args.hidden_size, args.output_size)
    model = DenseMLP(layer1, layer2).to(device)
    # Load saved weights.  Provide a helpful error message if loading fails.
    try:
        state_dict = torch.load(args.model_path, map_location=device)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model from {args.model_path}. Ensure that the file "
            "exists and contains a valid PyTorch state dictionary."
        ) from e
    model.load_state_dict(state_dict)
    model.eval()
    loader = get_test_loader(args.batch_size)
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            if data.dim() > 2:
                data = data.view(data.size(0), -1)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    accuracy = 100.0 * correct / len(loader.dataset)
    print(f"Test accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
