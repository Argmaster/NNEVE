from matplotlib import pyplot as plt

from nneve.quantum_oscillator.examples import default_qo_network

# acquire network object with precalculated weights
# for quantum oscillator state 1 (base)
network = default_qo_network(state=1)
network.plot_solution()
plt.plot()
