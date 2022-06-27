from pathlib import Path

from ..quantum_oscilator import QOConstants, QONetwork, QOTracker

__all__ = ["default_qo_network"]

DIR = Path(__file__).parent


def default_qo_network(state: int = 1) -> QONetwork:
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
    network = QONetwork(constants=constants, is_debug=True)
    network.load_weights(DIR / "weights" / f"s{state}.w")
    return network
