#!/usr/bin/env python3

# -----------------------------------------------------------------------------
# Project Name: Neural Splines
# File: train.py
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
train.py
========

Training script for the neural spline network on the MNIST dataset.

This script constructs a simple multilayer perceptron using the
compressed spline layers defined in ``neural_spline.py``.  It
demonstrates that spline‑based weight representations are capable of
learning a meaningful model despite having far fewer parameters than
their dense counterparts.

This training routine is described in Section~4 (Training on MNIST)
of the accompanying LaTeX document.  For background on the spline
layers themselves see Section~2.

At the end of training the script converts the spline network into a
dense equivalent and writes two checkpoint files: one containing the
original spline network and another containing the densified model.
The densified model is exportable to ExecuTorch because it no longer
contains interpolation operations.

Usage:
    python3 train.py --epochs 5 --batch-size 128 --cp 6

By default training runs for a small number of epochs to keep
examples lightweight.  Feel free to adjust hyper‑parameters for
better accuracy.
"""

import argparse
import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from neural_spline import SplineMLP


def get_data_loaders(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Return training and test data loaders for MNIST.

    Parameters
    ----------
    batch_size : int
        Number of images per batch.

    Returns
    -------
    (DataLoader, DataLoader)
        Tuple containing the training and test loaders.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    try:
        train_set = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_set = datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )
    except Exception as e:
        # If the dataset cannot be downloaded (e.g. due to network issues),
        # raise a more informative error.  See Section~4 of the LaTeX
        # document for discussion of the MNIST training pipeline.
        raise RuntimeError(
            "Failed to download or load the MNIST dataset. Please ensure "
            "internet connectivity or provide a pre‑downloaded dataset in ./data."
        ) from e
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    epoch: int,
) -> None:
    model.train()
    total_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]"
                f"\tLoss: {loss.item():.6f}"
            )
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} average loss: {avg_loss:.6f}")


def test(model: nn.Module, device: torch.device, test_loader: DataLoader) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(f"Test accuracy: {accuracy:.2f}%")
    return accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a neural spline MLP on MNIST")
    parser.add_argument("--epochs", type=int, default=5, help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="training batch size")
    parser.add_argument(
        "--cp",
        type=int,
        default=4,
        help="number of control points per dimension in spline layers",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
        help="number of hidden units in the MLP",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints",
        help="directory to save the trained models",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="learning rate"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_data_loaders(args.batch_size)

    model = SplineMLP(28 * 28, args.hidden_size, 10, args.cp, args.cp)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        test(model, device, test_loader)

    # Save spline model checkpoint
    spline_path = os.path.join(args.output_dir, "spline_model.pth")
    torch.save(model.state_dict(), spline_path)
    print(f"Saved spline model to {spline_path}")

    # Convert to dense model and evaluate again
    dense_model = model.to_dense_mlp().to(device)
    test(dense_model, device, test_loader)
    dense_path = os.path.join(args.output_dir, "dense_model.pth")
    torch.save(dense_model.state_dict(), dense_path)
    print(f"Saved dense model to {dense_path}")


if __name__ == "__main__":
    main()