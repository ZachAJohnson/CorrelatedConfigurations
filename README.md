# CorrelatedConfigurations

A small Python package for generating spatial configurations and solving integral equation theory (IET) problems in systems with Yukawa (and similar) interactions. The package is designed for researchers and developers working on simulation and analysis of complex fluids or plasmas.

## Features

- **Configuration Generation**
  - Partition a simulation cell into subcells.
  - Place particles using user-defined probability distributions.
  - Handle periodic boundary conditions.
- **Integral Equation Theory (IET) Solver**
  - Solve multi-component integral equations using closures (e.g., Hypernetted-Chain (HNC), Percus-Yevick (PY)).
  - Compute direct and total correlation functions.
  - Fourier transform routines to convert between real- and reciprocal-space.
- **Utilities and Constants**
  - Contains modules for physical constants and helper functions.
- **Examples**
  - A Jupyter Notebook example demonstrating a Yukawa system is provided in the `examples` folder.

## Installation

Clone the repository and install the package in editable mode:

```bash
git clone git@github.com:ZachAJohnson/CorrelatedConfigurations.git
cd CorrelatedConfigurations
pip install -e .
