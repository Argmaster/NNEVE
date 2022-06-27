# NNEVE

Neural network based eigenvalue estimator for quantum oscillator problem.

# Installation

To install this project from PyPI use following command

```bash
pip install nneve-project
```

# Quick start

To view quantum oscillator approximation for states 1 to 7 you can use
precalculated weights:

```python
from nneve.examples import default_qo_network

# network with weights for state 1
network = default_qo_network(state=1)

```

To manually run learning cycle check out
["How to run QONetwork learning cycle"](/quantum_oscilator/learning_cycle/) in
Quantum Oscillator section of docs.

# Documentation

Online documentation is available at
[argmaster.github.io/NNEVE/](https://argmaster.github.io/NNEVE/)

To build docs locally run

```
tox -e docs
```
