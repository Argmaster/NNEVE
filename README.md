# NNEVE

NNEVE is a collection of neural network based solutions to physics based
problems. As for now only network for quantum oscillator approximation is fully
implemented. Hopefully soon will arrive neural network for solving
Navier-Stokes equation based on limited number of measurement points..

# Installation

This project is uploaded to PyPI as `nneve`, therefore can be installed with
following command

```bash
pip install nneve
```

At least Python 3.7 is required.

# Quick example

To view quantum oscillator approximation for states 1 to 7 you can load
precalculated weights and acquire model object with following snippet:

```python
from nneve.examples import default_qo_network

# network with weights for state 1
network = default_qo_network(state=1)
network.plot_solution()

```

To manually run learning cycle check out
["How to run QONetwork learning cycle"](https://argmaster.github.io/NNEVE//quantum_oscillator/learning_cycle/)
in Quantum Oscillator section of docs.

# Documentation

Online documentation is available at
[argmaster.github.io/NNEVE/](https://argmaster.github.io/NNEVE/)

To build docs locally run

```
tox -e docs
```
