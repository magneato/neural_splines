# Neural Splines for Resource‑Constrained Deep Learning

This repository presents a self‑contained example of training and
deploying a compressed neural network using spline interpolation.
Neural networks typically store one parameter per connection, which
quickly becomes burdensome on devices with limited memory.  A **neural
spline** represents an entire weight matrix with a much smaller grid
of control points and reconstructs the dense weights by bicubic
interpolation on the fly.  Fewer parameters mean smaller models and
lower memory footprints while maintaining competitive accuracy.

The implementation is deliberately simple and educational.  All code
uses vanilla PyTorch and runs on CPU or GPU without any special
runtime.  A LaTeX companion in `docs/main.tex` offers a concise
introduction to the theory of spline‑based compression and explains
how the provided code works.  This project is intended as a
masterclass for researchers in resource‑constrained environments
seeking to understand and extend neural splines.

## Overview

The project comprises two stages:

1. **Training a compressed model** on the MNIST digit dataset using
   spline layers (see `src/train.py`).  During training the network
   learns a small set of control points and compares its performance
   against a densified version of itself.
2. **Densifying and running inference** with standard PyTorch.  After
   training the spline network is converted to an equivalent dense
   model and saved.  The script `src/inference.py` loads this dense
   model and evaluates it on the MNIST test set.

## Getting Started

### Prerequisites

* Ubuntu 24.04 (other UNIX‑like systems may work with minor changes).
* A Python 3.8+ interpreter.  The provided `setup.sh` script installs
  Python via the system package manager if needed.
* Internet access to download PyTorch, torchvision and the MNIST dataset.

### Setup

Run the provided shell script from the repository root to install
system packages and create a virtual environment:

```bash
bash setup.sh
```

The script installs PyTorch and torchvision via `pip`.  The versions
are pinned for reproducibility; feel free to adjust them.  After
execution you can activate the environment via

```bash
source .venv/bin/activate
```

### Training

To train the spline network on MNIST run:

```bash
python3 src/train.py --epochs 5 --batch-size 128 --cp 6 --hidden-size 256
```

The script downloads the MNIST dataset, constructs a network with
spline layers, trains it and writes two checkpoint files in
`./checkpoints`:

* `spline_model.pth` – the raw spline model containing control
  points.
* `dense_model.pth` – a densified version of the model where the
  interpolation has been baked into fixed weights.

After training the script prints the accuracy of both the spline and
dense models.  Expect the compressed model to achieve around 95–97 %
accuracy after a few epochs.

### Inference

After training you can evaluate the densified model using the provided 
inference script.  The script reconstructs the dense architecture, loads 
the saved weights and reports the test accuracy:

```bash
python3 src/inference.py \
    --model-path checkpoints/dense_model.pth \
    --input-size 784 --hidden-size 256 --output-size 10
```

The `input-size`, `hidden-size` and `output-size` arguments should
match the architecture used during training.  In the MNIST example
the input is 28×28 pixels (784 features) and the output has 10
classes.

## Project Structure

```
neural_spline_project/
├── README.md                # This document
├── setup.sh                 # Environment setup script
├── docs/
│   └── main.tex             # LaTeX exposition of the theory and code
├── src/
│   ├── neural_spline.py     # Spline layer and MLP implementation
│   ├── train.py             # Training script for MNIST
│   └── inference.py         # Dense model inference script
└── checkpoints/ (created at runtime)
    ├── spline_model.pth
    └── dense_model.pth
```

## Philosophy and Future Work

While this repository focuses on practical code, the underlying
concept arises from a much broader epistemological perspective.  The
accompanying LaTeX document in `docs/main.tex` distils the ideas
presented in the original paper into a concise narrative that
bridges abstract notions of continuous parameter spaces with
implementable algorithms.  Readers are encouraged to explore the
theory to appreciate how neural splines can serve as a bridge between
 discrete computation and continuous knowledge representations.

Future improvements might include convolutional spline layers,
integration with quantization techniques and support for more
complex datasets.  Contributions are welcome.

## Acknowledgements

This project draws inspiration from a wide range of literature on
model compression and functional approximation.  It was prepared 
with care to empower researchers worldwide to experiment with 
compressed neural models and to encourage further exploration into
 transcendent parameter structures.