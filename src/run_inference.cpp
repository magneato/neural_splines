#if 0
# -----------------------------------------------------------------------------
# Project Name: Neural Splines
# File: run_inference.cpp
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
#endif

// run_inference.cpp
// -----------------------------------------------------------------------------
// A minimal C++ example that demonstrates how to load and run an ExecuTorch
// program (.pte file).  This file is intended for illustration only; the
// exact APIs may change between ExecuTorch releases.  Consult the official
// documentation for up‑to‑date information on the C++ runtime
// interfaces【627922175094803†L330-L464】.
//
// Usage:
//   ./run_inference path/to/model.pte
//
// The program constructs a Module from the provided .pte file,
// allocates an input tensor shaped (1,1,28,28) with zero values, and
// invokes the ``forward`` method.  The output tensor is printed to
// stdout.  To evaluate accuracy on a dataset, integrate this code
// with your preferred data loading routines and compare the argmax
// of the output with the ground truth labels.
//
// Note: The ExecuTorch C++ API is still under active development.
// Some calls shown here may not compile on future versions without
// modification.  When in doubt refer to the example code provided in
// the ExecuTorch repository and documentation.

#include <iostream>
#include <vector>
#include <string>

// ExecuTorch headers.  These paths assume that the ExecuTorch C++
// headers have been installed into the include directory discovered
// by CMake.  You may need to adjust these include statements if your
// installation differs.
#include <executorch/extension/module/module.h>
#include <executorch/runtime/runtime.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.pte>" << std::endl;
        return 1;
    }
    std::string model_path = argv[1];

    try {
        // Instantiate a Module from the .pte file.  The Module API
        // hides most of the low‑level details of the ExecuTorch
        // runtime【627922175094803†L330-L464】.
        ::executorch::extension::Module module(model_path.c_str());

        // Prepare a dummy input tensor.  In a real application you
        // would populate this with image data.  For MNIST the input
        // shape is (1,1,28,28).  The runtime defines its own Tensor
        // type, which is constructed from raw data, shape and data
        // type.  See the ExecuTorch runtime API reference for more
        // details.
        std::vector<int64_t> shape = {1, 1, 28, 28};
        std::vector<float> input_data(1 * 1 * 28 * 28, 0.0f);
        ::executorch::runtime::Tensor input_tensor(
            ::executorch::runtime::ScalarType::Float, shape, input_data.data());

        // Execute the forward method.  The Module API accepts a
        // vector of inputs and returns a vector of outputs.
        std::vector<::executorch::runtime::EValue> inputs;
        inputs.emplace_back(input_tensor);
        auto result = module.execute("forward", inputs);

        if (!result.ok()) {
            std::cerr << "Error executing model: "
                      << static_cast<int>(result.error()) << std::endl;
            return 1;
        }

        // Retrieve the output tensor.  The example model returns a
        // single tensor containing class logits.
        auto outputs = *result;
        if (outputs.empty() || !outputs[0].is_tensor()) {
            std::cerr << "Unexpected output format" << std::endl;
            return 1;
        }
        const auto& out_tensor = outputs[0].to_tensor();
        // Print the raw scores.  A production system would apply an
        // argmax over the last dimension to obtain the predicted
        // class.
        std::cout << "Output scores:";
        for (size_t i = 0; i < out_tensor.numel(); ++i) {
            std::cout << ' ' << out_tensor.data_ptr<float>()[i];
        }
        std::cout << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}