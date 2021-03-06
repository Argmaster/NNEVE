import gc
from pathlib import Path

import matplotlib
import tensorflow as tf
from matplotlib import pyplot as plt

from nneve.quantum_oscillator import (
    QOConstants,
    QONetwork,
    QOParams,
    QOTracker,
)
from nneve.utility.testing import disable_gpu_or_skip

EXAMPLES_CODE = Path(__file__).parent
EXAMPLES_DIR = EXAMPLES_CODE.parent
WEIGHTS_DIR = EXAMPLES_DIR / "weights"
PLOTS_DIR = EXAMPLES_DIR / "plots"


tf.random.set_seed(0)
disable_gpu_or_skip()

constants = QOConstants(
    k=4.0,
    mass=1.0,
    x_left=-6.0,
    x_right=6.0,
    fb=0.0,
    sample_size=1200,
    tracker=QOTracker(),
)
network = QONetwork(constants=constants, is_debug=True)

network.summary()
matplotlib.use("Agg")

for index, nn in enumerate(
    network.train_generations(
        QOParams(c=-2.0, c_step=0.16),
        generations=150,
        epochs=1000,
    )
):
    x = nn.constants.get_sample()
    y2, _ = nn(x)
    nn.constants.tracker.plot(y2, x)
    # savefig tends to create memory leaks
    plt.savefig(PLOTS_DIR / f"{index}.png")
    plt.cla()
    plt.clf()
    plt.close("all")
    gc.collect()
    nn.save(WEIGHTS_DIR / f"{index}.w")
