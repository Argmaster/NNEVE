from io import BytesIO
from pathlib import Path

import pytest
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image

from nneve.benchmark.plotting import get_image_identity_fraction
from nneve.quantum_oscilator.network import QOConstants, QONetwork
from nneve.quantum_oscilator.tracker import QOTracker

DIR = Path(__file__).parent
DATA_DIR = DIR / "data"


class TestQOTracker:
    @pytest.fixture()
    def network(self) -> QONetwork:
        tf.random.set_seed(0)
        # ; disable_gpu_or_skip()

        constants = QOConstants(
            k=4.0,
            mass=1.0,
            x_left=-6.0,
            x_right=6.0,
            fb=0.0,
            sample_size=1200,
            tracker=QOTracker(),
            neuron_count=50,
        )
        return QONetwork(constants=constants, is_debug=True)

    # @pytest.mark.skip("Not functional due to size mismatch")
    def test_load_network(self, network: QONetwork) -> None:  # noqa: FNE004
        network.load(DATA_DIR / "example.w")
        plt.clf()
        plt.figure(figsize=(10, 10), dpi=100)
        network.plot_solution()

        buffer = BytesIO()
        plt.savefig(buffer)
        buffer.seek(0)

        compare = Image.open(DATA_DIR / "example.png")
        current = Image.open(buffer)

        assert get_image_identity_fraction(compare, current) > 0.99
