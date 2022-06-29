import pytest

from nneve.quantum_oscillator import QONetwork
from nneve.quantum_oscillator.examples import default_qo_network


@pytest.mark.parametrize("state", range(1, 8))
def test_qonetwork_load_from_file(state: int) -> None:
    network = default_qo_network(state)
    assert isinstance(network, QONetwork)
