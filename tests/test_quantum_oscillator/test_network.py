from io import BytesIO
from pathlib import Path

import pytest
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image

from nneve.quantum_oscillator import QOConstants, QONetwork, QOTracker
from nneve.utility import disable_gpu_or_skip, get_image_identity_fraction

DIR = Path(__file__).parent
DATA_DIR = DIR / "data"
SAMPLE_SIZE: int = 32


class TestQOTracker:
    @pytest.fixture()
    def network(self) -> QONetwork:
        tf.random.set_seed(0)
        disable_gpu_or_skip()

        constants = QOConstants(
            k=4.0,
            mass=1.0,
            x_left=-6.0,
            x_right=6.0,
            fb=0.0,
            sample_size=SAMPLE_SIZE,
            tracker=QOTracker(),
        )
        return QONetwork(constants=constants, is_debug=True)

    def test_boundary_function_io(self, network: QONetwork) -> None:
        x = network.constants.get_sample()

        assert x.shape == (SAMPLE_SIZE, 1)
        assert x.dtype == tf.float32

        retval = network._boundary_function(x)

        assert isinstance(retval, tf.Tensor)
        assert retval.shape == (SAMPLE_SIZE, 1)
        assert retval.dtype == tf.float32

    def test_potential_function_io(self, network: QONetwork) -> None:
        x = network.constants.get_sample()

        assert x.shape == (SAMPLE_SIZE, 1)
        assert x.dtype == tf.float32

        retval = network._potential_function(x)

        assert isinstance(retval, tf.Tensor)
        assert retval.shape == (SAMPLE_SIZE, 1)
        assert retval.dtype == tf.float32

    def test_parametric_solution_io(self, network: QONetwork) -> None:
        x = network.constants.get_sample()

        assert x.shape == (SAMPLE_SIZE, 1)
        assert x.dtype == tf.float32

        deriv_x = tf.Variable(initial_value=x)
        assert x.shape == deriv_x.shape == (SAMPLE_SIZE, 1)

        loss, eigenvalue = network._parametric_solution_function(deriv_x)  # type: ignore

        assert isinstance(loss, tf.Tensor)
        assert isinstance(eigenvalue, tf.Tensor)
        assert loss.shape == (SAMPLE_SIZE, 1)
        assert eigenvalue.shape == (1,)

    def test_load_network_from_file(
        self, network: QONetwork
    ) -> None:  # noqa: FNE004
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
