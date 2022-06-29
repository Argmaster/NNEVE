from typing import List, cast

import pytest
import tensorflow as tf

__all__ = [
    "disable_gpu_or_skip",
    "skip_if_no_gpu",
    "is_gpu",
    "disable_gpu",
]


def disable_gpu() -> None:  # pragma: no cover
    """Force tensorflow to ignore GPU for calculations."""
    tf.config.set_visible_devices([], "GPU")


def disable_gpu_or_skip() -> None:  # pragma: no cover
    """Force tensorflow to ignore GPU, then raise Skipped() to skip pytest test
    if attempt to disable GPU fails."""
    try:
        disable_gpu()
        assert tf.config.get_visible_devices("GPU") == []
    except Exception:
        pytest.skip("Failed to disable GPU.")


def skip_if_no_gpu() -> None:  # pragma: no cover
    """Skips pytest test when there is not GPU available for calculations."""
    if not is_gpu():
        pytest.skip("No GPU available for testing.")


def is_gpu() -> bool:
    return len(cast(List, tf.config.get_visible_devices("GPU"))) > 0
