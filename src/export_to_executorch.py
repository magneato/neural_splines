#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Project Name: Neural Splines
# File: export_to_executorch.py
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
export_to_executorch.py
=======================

This script converts a trained dense PyTorch model into an ExecuTorch
program (.pte file) for deployment on edge devices.  It relies on
ExecuTorch’s export APIs, which use the standard PyTorch
``torch.export`` flow followed by backend specific lowering.  See
the official ExecuTorch documentation for additional details and
supported backends.

Example usage:

    python3 export_to_executorch.py \
        --model-path checkpoints/dense_model.pth \
        --out-pte checkpoints/mnist_model.pte

The resulting .pte file can be executed on device using the
ExecuTorch runtime or loaded from Python via
``executorch.runtime.Runtime``.

Note
----
The ExecuTorch Python package must be installed.  Installation
instructions are provided in the ``setup.sh`` script and in the
official documentation【748237474359461†L204-L217】.  Exporting also requires PyTorch
2.2 or later because it uses ``torch.export``.
"""

import argparse
import os

import torch

from neural_spline import DenseMLP

try:
    # Import ExecuTorch export helpers.  If this import fails it
    # likely means that the ExecuTorch package is not installed.
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackPartitioner,
    )
    from executorch.exir import to_edge_transform_and_lower
except ImportError as e:
    raise SystemExit(
        "Failed to import ExecuTorch. Ensure executorch is installed via pip."
    ) from e


def load_dense_model(model_path: str) -> DenseMLP:
    """Instantiate a dense MLP and load its parameters from a file.

    The network architecture must match that used during training
    (784→hidden→10).  If your model uses a different hidden size you
    may need to adjust this function accordingly.

    Parameters
    ----------
    model_path : str
        Path to the checkpoint file containing the dense model
        weights.

    Returns
    -------
    DenseMLP
        A dense MLP with loaded parameters ready for export.
    """
    # The hidden size is encoded in the state dict of the dense
    # model.  We infer it by reading the shape of the first weight.
    state_dict = torch.load(model_path, map_location="cpu")
    # Weight names follow the pattern layer1.weight for the first
    # linear layer and layer2.weight for the second.  The shape
    # [hidden_size, input_size] gives us the hidden size.
    w1 = state_dict["layer1.weight"]
    input_size = w1.shape[1]
    hidden_size = w1.shape[0]
    output_size = state_dict["layer2.weight"].shape[0]
    model = DenseMLP(
        layer1=torch.nn.Linear(input_size, hidden_size),
        layer2=torch.nn.Linear(hidden_size, output_size),
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def export_model(
    model: torch.nn.Module, sample_input: torch.Tensor, out_pte: str
) -> None:
    """Export a PyTorch model to an ExecuTorch program.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to export.  It must be exportable via
        ``torch.export.export`` (no unsupported control flow or
        operators).
    sample_input : torch.Tensor
        Example input tensor used to trace the model.  The shape
        should reflect the expected input signature for inference.
    out_pte : str
        Path where the resulting `.pte` file will be written.
    """
    # First convert the model to an Export IR.  ``torch.export``
    # captures the computation graph of the model given example
    # inputs.  See the ExecuTorch docs for details【748237474359461†L264-L279】.
    exported = torch.export.export(model, (sample_input,), {})
    # Lower to ExecuTorch format.  The partitioner selects a
    # hardware backend; here we use XNNPACK which targets generic
    # CPU architectures【748237474359461†L248-L253】.
    et_program = to_edge_transform_and_lower(
        exported, partitioner=[XnnpackPartitioner()]
    ).to_executorch()
    # Save the program to disk.  The .pte suffix is conventional.
    with open(out_pte, "wb") as f:
        f.write(et_program.buffer)
    print(f"Wrote ExecuTorch program to {out_pte}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a dense MNIST model to ExecuTorch")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="path to the dense model checkpoint (.pth) to export",
    )
    parser.add_argument(
        "--out-pte",
        type=str,
        required=True,
        help="output path for the generated ExecuTorch program (.pte)",
    )
    args = parser.parse_args()

    model = load_dense_model(args.model_path)
    # Create an example input.  ExecuTorch models accept the same
    # inputs as their PyTorch counterparts.  Our MLP flattens 28×28
    # images internally, so we can supply a 4‑D tensor shaped like an
    # image (batch, channels, height, width).  Using a float32 tensor
    # ensures compatibility with the default operator set.
    sample_input = torch.zeros((1, 1, 28, 28), dtype=torch.float32)
    export_model(model, sample_input, args.out_pte)


if __name__ == "__main__":
    main()