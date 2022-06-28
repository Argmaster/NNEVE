import gc
from io import BytesIO
from pathlib import Path

import pytest
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image

from nneve.quantum_oscilator.network import QOConstants, QONetwork
from nneve.quantum_oscilator.params import QOParams
from nneve.quantum_oscilator.tracker import QOTracker
from nneve.utility import disable_gpu_or_skip, get_image_identity_fraction

DIR = Path(__file__).parent
DATA_DIR = DIR / "data"
DATA_DIR.mkdir(0o777, True, True)


class TestQOTracker:
    def test_create_tracker(self) -> None:
        tracker = QOTracker()
        assert tracker.eigenvalue == []
        assert tracker.residuum == []
        assert tracker.drive_loss == []
        assert tracker.function_loss == []
        assert tracker.eigenvalue_loss == []
        assert tracker.c == []
        assert tracker.total_loss == []

    def test_ensure_new_lists(self) -> None:
        tracker1 = QOTracker()
        tracker2 = QOTracker()
        assert tracker1.eigenvalue is not tracker2.eigenvalue
        assert tracker1.residuum is not tracker2.residuum
        assert tracker1.drive_loss is not tracker2.drive_loss
        assert tracker1.function_loss is not tracker2.function_loss
        assert tracker1.eigenvalue_loss is not tracker2.eigenvalue_loss
        assert tracker1.c is not tracker2.c
        assert tracker1.total_loss is not tracker2.total_loss

    def test_push_stats(self) -> None:
        tracker = QOTracker()
        tracker.push_stats(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
        assert len(tracker.total_loss) == 1
        assert tracker.total_loss[0] == 1.0
        assert len(tracker.eigenvalue) == 1
        assert tracker.eigenvalue[0] == 2.0
        assert len(tracker.residuum) == 1
        assert tracker.residuum[0] == 3.0
        assert len(tracker.function_loss) == 1
        assert tracker.function_loss[0] == 4.0
        assert len(tracker.eigenvalue_loss) == 1
        assert tracker.eigenvalue_loss[0] == 5.0
        assert len(tracker.drive_loss) == 1
        assert tracker.drive_loss[0] == 6.0
        assert len(tracker.c) == 1
        assert tracker.c[0] == 7.0

    def test_get_trace(self) -> None:
        tracker = QOTracker()
        tracker.push_stats(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
        trace = tracker.get_trace(0)
        assert isinstance(trace, str)
        print(repr(trace))
        assert trace == "epoch: 1      loss: 1.0000     λ: 2.0000     c: 7.00 "

    @pytest.fixture()
    def network(self) -> QONetwork:
        tf.random.set_seed(0)
        disable_gpu_or_skip()
        return QONetwork(constants=QOConstants(), is_debug=True)

    def test_run_network_with_trace(self, network: QONetwork) -> None:
        network.summary()
        assert network.constants.tracker is not None
        for index, _ in enumerate(
            network.train_generations(
                QOParams(c=-2.0, c_step=0.48),
                generations=4,
                epochs=10,
            )
        ):
            assert (
                len(network.constants.tracker.eigenvalue) == (index + 1) * 10
            )

    def test_run_network_with_plot(self, network: QONetwork) -> None:
        network.summary()
        assert network.constants.tracker is not None
        for index, _ in enumerate(
            network.train_generations(
                QOParams(c=-2.0, c_step=0.48),
                generations=4,
                epochs=10,
            )
        ):
            x = network.constants.get_sample()
            y2, _ = network(x)
            network.constants.tracker.plot(y2, x)

            buffer = BytesIO()

            plt.savefig(buffer, dpi=100, format="png")
            # ; plt.savefig(DATA_DIR / f"plot_{index}.png", dpi=100, format="png")
            plt.cla()
            plt.clf()
            plt.close("all")
            gc.collect()

            buffer.seek(0)
            i1 = Image.open(DATA_DIR / f"plot_{index}.png")
            i2 = Image.open(buffer)
            assert get_image_identity_fraction(i1, i2) > 0.9
