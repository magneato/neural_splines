# -----------------------------------------------------------------------------
# Project Name: Neural Splines
# File: neural_spline.py
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
neural_spline.py
================

This module provides PyTorch components that implement weight
compression using two‑dimensional spline interpolation.  Traditional
dense layers store one parameter per connection, which quickly
becomes prohibitive on resource‑constrained hardware.  Neural
splines address this by representing the full weight matrix with a
much smaller grid of **control points**.  During the forward
pass the grid is up‑sampled using bicubic interpolation to
construct the dense weight matrix on the fly.  A separate vector of
control points defines the bias.

The implementation below is intentionally concise yet educational.
It depends only on the core PyTorch library and does not require
any external runtime.  After training you can convert a spline
layer into an equivalent dense layer so that inference can proceed
without on‑the‑fly interpolation.  Readers interested in the
mathematics of splines will find an overview in the accompanying
paper and LaTeX document.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class SplineLinear(nn.Module):
    """Linear layer with weights defined by a bicubic spline.

    Instead of storing a full ``out_features × in_features`` weight
    matrix, this layer maintains a much smaller set of control
    points.  During the forward pass the control point grid is
    interpolated to the required shape using bicubic interpolation
    provided by :func:`torch.nn.functional.interpolate`.

    A separate vector of control points is used to generate the
    bias.  This approach dramatically reduces the number of trainable
    parameters while still allowing the network to approximate the
    expressivity of a dense layer.

    This implementation follows the theoretical discussion in
    Section~2 (Spline--Based Compression) and Section~3 (Reference
    Implementation) of the accompanying LaTeX document.  For a detailed
    derivation of the interpolation formula see equation~(1) in
    Section~2.

    Parameters
    ----------
    in_features : int
        The dimensionality of the input.
    out_features : int
        The dimensionality of the output.
    cp_h : int, optional
        Number of control points along the output (height) axis.  A
        larger value allows the spline to model finer variation along
        the output dimension.  Defaults to 4.
    cp_w : int, optional
        Number of control points along the input (width) axis.
        Defaults to 4.
    cp_bias : int, optional
        Number of control points used to generate the bias.  If
        omitted, ``cp_bias`` defaults to the same value as
        ``cp_h``.
    """


    def __init__(
        self,
        in_features: int,
        out_features: int,
        cp_h: int = 4,
        cp_w: int = 4,
        cp_bias: int | None = None,
    ) -> None:
        """
        Initialize a spline‐linear layer.

        Parameters
        ----------
        in_features : int
            Number of input features.
        out_features : int
            Number of output features.
        cp_h : int, default=4
            Number of control points along the output dimension (height).
        cp_w : int, default=4
            Number of control points along the input dimension (width).
        cp_bias : int, optional
            Number of control points for the bias spline.  If omitted,
            defaults to ``cp_h``.

        Notes
        -----
        Cubic B‑spline interpolation requires at least four control points in
        each dimension.  If ``cp_h`` or ``cp_w`` are less than 4 a
        ``ValueError`` is raised.  See Section \ref{sec:spline-compression}
        of the accompanying LaTeX document for the mathematical
        formulation of the interpolation.
        """
        super().__init__()
        if cp_h < 4 or cp_w < 4:
            raise ValueError(
                "cp_h and cp_w must be at least 4 for cubic spline interpolation"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.cp_h = cp_h
        self.cp_w = cp_w
        self.cp_bias = cp_bias if cp_bias is not None else cp_h

        # Weight control points.  Initialized with a small random
        # distribution to avoid large initial outputs.  Shape: (Hcp, Wcp)
        self.weight_control_points = nn.Parameter(
            torch.randn(self.cp_h, self.cp_w) * 0.02
        )

        # Bias control points.  Shape: (Bcp,)
        self.bias_control_points = nn.Parameter(
            torch.randn(self.cp_bias) * 0.02
        )

    def _interpolate_weights(self) -> torch.Tensor:
        """Generate the dense weight matrix by bicubic interpolation.

        Returns
        -------
        torch.Tensor
            A tensor of shape ``(out_features, in_features)``.
        """
        # Reshape control point grid to match the expected input of
        # torch.nn.functional.interpolate.  The function expects
        # input with shape (N, C, H, W).  We use N=C=1 because we
        # interpolate a single 2‑D surface.
        cp = self.weight_control_points.unsqueeze(0).unsqueeze(0)
        # Perform bicubic interpolation to the desired spatial size.
        # ``align_corners=True`` ensures that the extreme control
        # points influence the extreme weights exactly.
        dense = F.interpolate(
            cp,
            size=(self.out_features, self.in_features),
            mode="bicubic",
            align_corners=True,
        )
        # Remove batch and channel dimensions.
        return dense.squeeze(0).squeeze(0)

    def _interpolate_bias(self) -> torch.Tensor:
        """Generate the dense bias vector by linear interpolation.

        Returns
        -------
        torch.Tensor
            A tensor of shape ``(out_features,)``.
        """
        # For a 1‑D bias we treat the control points as a length
        # ``cp_bias`` sequence.  By adding dummy dimensions we can
        # reuse :func:`torch.nn.functional.interpolate` for linear
        # interpolation.  The input shape becomes (N, C, L), and the
        # output will be (N, C, L_out).  We again set N=C=1.
        cp = self.bias_control_points.unsqueeze(0).unsqueeze(0)
        dense = F.interpolate(
            cp,
            size=(self.out_features),
            mode="linear",
            align_corners=True,
        )
        return dense.squeeze(0).squeeze(0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Compute the linear transformation with spline‑generated weights.

        Parameters
        ----------
        input : torch.Tensor
            A tensor of shape ``(batch_size, in_features)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(batch_size, out_features)``.
        """
        weight = self._interpolate_weights()
        bias = self._interpolate_bias()
        return F.linear(input, weight, bias)

    def to_dense_linear(self) -> nn.Linear:
        """Return an equivalent dense :class:`~torch.nn.Linear` layer.

        Converting a spline layer to a dense layer fixes the
        interpolated weights and biases at their current values.
        This is useful for inference when you no longer need to
        update the control points and want to avoid the overhead of
        interpolation on every forward pass.

        Returns
        -------
        nn.Linear
            A dense linear layer with identical output to this
            spline layer at the current state of the control points.
        """
        dense_weight = self._interpolate_weights()
        dense_bias = self._interpolate_bias()
        linear = nn.Linear(self.in_features, self.out_features)
        # Copy the tensors into the new layer without tracking
        # gradients.  The ``detach()`` call prevents PyTorch from
        # storing a backward reference to the original control
        # points.  ``clone()`` ensures the data is copied.
        linear.weight.data = dense_weight.detach().clone()
        linear.bias.data = dense_bias.detach().clone()
        return linear


class SplineMLP(nn.Module):
    """A simple multilayer perceptron using spline linear layers.

    The network contains two spline linear layers separated by a
    rectified linear unit.  It is designed for small image
    classification tasks such as MNIST, where the input is flattened
    into a one‑dimensional vector.  By adjusting the number of
    control points you can trade expressivity for parameter count.

    This architecture corresponds to the reference implementation
    described in Section~3 of the LaTeX document.  See Section~4
    (Training on MNIST) for usage and empirical results.

    Parameters
    ----------
    input_size : int
        Number of input features.  For 28×28 MNIST images this is
        784.
    hidden_size : int
        Number of hidden units in the intermediate representation.
    output_size : int
        Number of output units (e.g. 10 for MNIST digit classes).
    cp_hidden : int, optional
        Number of control points along each dimension for the
        hidden layer.  Defaults to 4.
    cp_output : int, optional
        Number of control points along each dimension for the
        output layer.  Defaults to 4.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        cp_hidden: int = 4,
        cp_output: int = 4,
    ) -> None:
        super().__init__()
        self.spline1 = SplineLinear(input_size, hidden_size, cp_hidden, cp_hidden)
        self.spline2 = SplineLinear(hidden_size, output_size, cp_output, cp_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten the input if it has more than two dimensions
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = self.spline1(x)
        x = F.relu(x)
        x = self.spline2(x)
        return x

    def to_dense_mlp(self) -> nn.Module:
        """Convert the spline MLP into an equivalent dense MLP.

        The returned network consists of ordinary
        :class:`~torch.nn.Linear` layers whose weights and biases are
        fixed to the current interpolated values of the spline
        network.  This conversion eliminates interpolation overhead
        during inference and decouples the dense model from the
        control points.

        Returns
        -------
        nn.Module
            A dense neural network with the same architecture and
            output as the original spline MLP.
        """
        dense1 = self.spline1.to_dense_linear()
        dense2 = self.spline2.to_dense_linear()
        return DenseMLP(dense1, dense2)


class DenseMLP(nn.Module):
    """A dense multilayer perceptron with fixed weights.

    This class mirrors the architecture of :class:`~SplineMLP` but
    uses standard dense layers.  It is used after training a
    :class:`~SplineMLP` to provide a version of the network that can
    be exported and executed on devices that do not support spline
    interpolation at runtime.

    The motivation for this class is discussed in Section~6
    (Densification and Inference) of the LaTeX document.  After
    densification the network no longer references control points or
    interpolation operations, simplifying inference.
    """

    def __init__(self, layer1: nn.Linear, layer2: nn.Linear) -> None:
        super().__init__()
        self.layer1 = layer1
        self.layer2 = layer2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        return x