import pytest
import tensorflow as tf

from nneve.quantum_oscilator.network import QOConstants, QONetwork
from nneve.quantum_oscilator.params import QOParams
from nneve.quantum_oscilator.tracker import QOTracker


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

    def test_plotting(self, network: QONetwork) -> None:
        network.summary()
        for _ in enumerate(
            network.train_generations(
                QOParams(c=-2.0, c_step=0.48),
                generations=4,
                epochs=10,
            )
        ):
            pass
