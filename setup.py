#!/usr/bin/env python3
"""
Neural Splines: Transform neural networks into interpretable mathematical curves
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages

# Ensure we're using the right Python version
if sys.version_info < (3, 8):
    sys.exit("Python 3.8 or higher is required")

# Read the README file
def read_readme():
    """Read README.md for long description"""
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Neural Splines: Transform neural networks into interpretable mathematical curves"

# Read requirements from file
def read_requirements(filename):
    """Read requirements from requirements file"""
    req_path = Path(__file__).parent / filename
    if req_path.exists():
        with open(req_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []

# Core dependencies
INSTALL_REQUIRES = [
    "torch>=2.1.0",
    "numpy>=1.21.0", 
    "scipy>=1.7.0",
    "matplotlib>=3.4.0",
    "tqdm>=4.62.0",
    "safetensors>=0.3.0",
    "transformers>=4.30.0",
    "accelerate>=0.20.0",
    "huggingface-hub>=0.16.0",
    "fbgemm-gpu"
]

# Optional dependencies
EXTRAS_REQUIRE = {
    "dev": [
        "pytest>=6.2.0",
        "pytest-cov>=2.12.0", 
        "black>=21.6.0",
        "isort>=5.9.0",
        "flake8>=3.9.0",
        "mypy>=0.910",
        "pre-commit>=2.13.0"
    ],
    "visualization": [
        "seaborn>=0.11.0",
        "plotly>=5.0.0", 
        "ipython>=7.25.0",
        "jupyter>=1.0.0"
    ],
    "deepseek": [
        "tokenizers>=0.13.0",
        "datasets>=2.0.0"
    ]
}

# Add 'full' option that includes everything
EXTRAS_REQUIRE["full"] = list(set(
    dep for deps in EXTRAS_REQUIRE.values() for dep in deps
))

# Entry points for command-line tools
ENTRY_POINTS = {
    "console_scripts": [
        "neural-splines=neural_splines.cli:main",
        "spline-convert=neural_splines.convert:cli_main", 
        "spline-visualize=neural_splines.visualization:cli_main"
    ]
}

# Package data
PACKAGE_DATA = {
    "neural_splines": [
        "data/*.json",
        "configs/*.yaml", 
        "templates/*.jinja2"
    ]
}

# Classifiers for PyPI
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research", 
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9", 
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Software Development :: Libraries :: Python Modules"
]

def get_version():
    """Get version from package"""
    try:
        # Try to read from neural_splines/__init__.py
        init_file = Path(__file__).parent / "neural_splines" / "__init__.py"
        if init_file.exists():
            with open(init_file, "r") as f:
                for line in f:
                    if line.startswith("__version__"):
                        return line.split("=")[1].strip().strip('"').strip("'")
    except Exception:
        pass
    
    # Default version
    return "1.0.0"

def main():
    """Main setup function"""
    setup(
        name="neural-splines",
        version=get_version(),
        description="Transform neural networks into interpretable mathematical curves using harmonic decomposition and spline interpolation",
        long_description=read_readme(),
        long_description_content_type="text/markdown",
        author="Neural Splines Project",
        author_email="neural-splines@example.com",
        url="https://github.com/your-username/neural-splines",
        project_urls={
            "Homepage": "https://github.com/your-username/neural-splines",
            "Repository": "https://github.com/your-username/neural-splines", 
            "Documentation": "https://neural-splines.readthedocs.io",
            "Bug Tracker": "https://github.com/your-username/neural-splines/issues"
        },
        packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
        package_data=PACKAGE_DATA,
        include_package_data=True,
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRAS_REQUIRE,
        entry_points=ENTRY_POINTS,
        python_requires=">=3.8",
        classifiers=CLASSIFIERS,
        keywords=[
            "neural-networks", 
            "splines", 
            "model-compression", 
            "interpretable-ai",
            "harmonic-decomposition", 
            "geometric-ml",
            "parameter-efficiency"
        ],
        license="MIT",
        zip_safe=False,
        
        # Additional metadata
        platforms=["any"],
        
        # For development installations
        cmdclass={},
        
        # Data files (if needed)
        data_files=[],
    )

if __name__ == "__main__":
    main()
