import tensorflow as tf

from nneve.benchmark.testing import disable_gpu_or_skip
from nneve.quantum_oscilator.network import QOConstants, QONetwork
from nneve.quantum_oscilator.tracker import QOTracker


def test_validate_output():
    tf.random.set_seed(0)
    disable_gpu_or_skip()

    constants = QOConstants(
        k=4.0,
        mass=1.0,
        x_left=-6.0,
        x_right=6.0,
        fb=0.0,
        sample_size=16,
        tracker=QOTracker(),
    )
    nn = QONetwork(constants=constants, is_debug=True)
    x = constants.sample()
    deriv_x = tf.Variable(initial_value=x)
    assert x.shape == deriv_x.shape == (16, 1)
    retval = nn._parametric_solutions_function(deriv_x)
    assert retval.shape == (16, 1)
