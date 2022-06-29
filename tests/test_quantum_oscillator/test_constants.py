import typing

from tensorflow import keras

from nneve.quantum_oscillator import QOConstants, QOTracker

if typing.TYPE_CHECKING:
    from keras.api._v2 import keras  # noqa: F811 # pragma: no cover


class TestQOConstants:
    def test_default_construct(self):
        const = QOConstants()
        assert isinstance(const.optimizer, keras.optimizers.Adam)
        assert isinstance(const.tracker, QOTracker)
        assert const.k == 4.0
        assert const.mass == 1.0
        assert const.x_left == -6.0
        assert const.x_right == 6.0
        assert const.fb == 0.0
        assert const.sample_size == 1000
        assert const.v_f == 1.0
        assert const.v_lambda == 1.0
        assert const.v_drive == 1.0

    def test_manual_construct(self):
        const = QOConstants(
            optimizer=keras.optimizers.SGD(),
            tracker=QOTracker(),
            k=1.0,
            mass=3.0,
            x_left=-4.0,
            x_right=4.0,
            fb=1.0,
            sample_size=500,
            v_f=3.23,
            v_lambda=3.23,
            v_drive=3.23,
        )
        assert isinstance(const.optimizer, keras.optimizers.SGD)
        assert isinstance(const.tracker, QOTracker)
        assert const.k == 1.0
        assert const.mass == 3.0
        assert const.x_left == -4.0
        assert const.x_right == 4.0
        assert const.fb == 1.0
        assert const.sample_size == 500
        assert const.v_f == 3.23
        assert const.v_lambda == 3.23
        assert const.v_drive == 3.23

    def test_create_sample(self):
        const = QOConstants(
            tracker=QOTracker(),
            x_left=-4.0,
            x_right=4.0,
            sample_size=500,
        )
        assert const.get_sample().shape == (500, 1)
        assert min(const.get_sample()) == -4.0
        assert max(const.get_sample()) == 4.0
