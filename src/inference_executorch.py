#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Project Name: Neural Splines
# File: inference_executorch.py
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
inference_executorch.py
=======================

Run inference on the MNIST test set using a compiled ExecuTorch program.

This script uses the ExecuTorch runtime to load a `.pte` program and
execute it on a sequence of input tensors.  It demonstrates how
models exported via ``export_to_executorch.py`` can be validated
without relying on full PyTorch.  For each test example the script
computes the predicted digit and reports the overall accuracy.

Usage:
    python3 inference_executorch.py --model checkpoints/mnist_model.pte

References
----------
The runtime API used here is described in the ExecuTorch getting
started guide【748237474359461†L310-L319】.  The API consists of a
singleton ``Runtime`` object, a ``Program`` loaded from a .pte file
and a ``Method`` representing the exported function to execute.
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    from executorch.runtime import Runtime
except ImportError:
    raise SystemExit(
        "Failed to import executorch.runtime. Please install the executorch package."
    )


def get_test_loader(batch_size: int = 128) -> DataLoader:
    """Return a DataLoader for the MNIST test set.

    Only the test set is needed for inference.  The images are
    normalized to match the training transforms.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_set = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    return DataLoader(test_set, batch_size=batch_size, shuffle=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ExecuTorch inference on MNIST")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="path to the ExecuTorch program (.pte) to evaluate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="batch size for inference",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"ExecuTorch program not found: {args.model}")

    test_loader = get_test_loader(args.batch_size)

    # Initialize the ExecuTorch runtime and load the program.  See
    # the official documentation for details【748237474359461†L310-L319】.
    runtime = Runtime.get()
    program = runtime.load_program(args.model)
    # The exported model exposes a 'forward' method.
    method = program.load_method("forward")

    correct = 0
    total = 0
    for data, target in test_loader:
        # ExecuTorch expects a list of tensors as input.  Flatten
        # images to 4‑D shape (batch, channels, height, width).  We
        # detach from any computation graph and convert to the CPU.
        input_tensor = data.to(torch.float32)
        # ExecuTorch runtime currently operates on Torch tensors.
        outputs = method.execute([input_tensor])[0]
        # outputs is a list containing a single tensor.  Convert to
        # PyTorch tensor to compute argmax.
        preds = outputs.argmax(dim=1)
        correct += preds.eq(target).sum().item()
        total += target.size(0)

    accuracy = 100.0 * correct / total
    print(f"ExecuTorch inference accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()