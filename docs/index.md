<figure markdown>
  ![NNEVE](img/nneve.jpg){ width="1000" }
</figure>

NNEVE is a collection of neural network based solutions to physics based
problems. As for now only network for quantum oscillator approximation is fully
implemented. Hopefully soon will arrive neural network for solving
Navier-Stokes equation based on limited number of measurement points..

# Installation

This project is uploaded to PyPI as `nneve`, therefore can be installed with
pip install (`Python 3.7` or newer is required)

[comment]: <> (https://github.com/ines/termynal)

<div class="termynal-block" id="termynal1" data-termynal>
    <span data-ty="input">pip install nneve</span>
    <span data-ty>Collecting nneve</span>
    <span data-ty style="padding-left:1rem;">Downloading nneve-22.6.28-py3-none-any.whl (20 kB)</span>
    <span data-ty="progress" style="padding-left:1rem;"></span>
    <span data-ty>Successfully installed nneve-22.6.28</span>
    <span data-ty></span>
</div>
<br/>

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
["How to run QONetwork learning cycle"](/quantum_oscillator/learning_cycle/) in
Quantum Oscillator section of docs.

# Documentation

Online documentation is available at
[argmaster.github.io/NNEVE/](https://argmaster.github.io/NNEVE/)

Builing docs locally is possible and well automated using tox virtual
environments. To be able to build documentation you have to acquire
`Python==3.8` and `tox>=3.24`, then you will be able to build docs

<div class="termynal-block" id="termynal2" data-termynal>
    <span data-ty="input">tox -e docs</span>
    <span data-ty>docs create: ~/repos/nneve/.tox/docs</span>
    <span data-ty>docs installdeps: -rrequirements-docs.txt</span>
    <span data-ty>...</span>
    <span data-ty>docs run-test: commands[0] | mkdocs build</span>
    <span data-ty>INFO - <span style="color:lime;">[macros] - Macros arguments:</span>...</span>
    <span data-ty>INFO - Documentation built in 0.49 seconds</span>
    <span data-ty style="padding-left:1rem;color:lime;">
        docs: commands succeeded
    </span>
    <span data-ty style="padding-left:1rem;color:lime;">
        congratulations :)
    </span>
    <span data-ty></span>
</div>
