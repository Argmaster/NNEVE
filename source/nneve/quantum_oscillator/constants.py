import sys
import typing
from typing import Sequence, cast

import tensorflow as tf
from pydantic import Field
from tensorflow import keras

from nneve.common import Model

from .tracker import QOTracker

if sys.version_info < (3, 8):
    from backports.cached_property import (  # noqa # pragma: no cover
        cached_property,
    )
else:
    from functools import cached_property  # noqa # pragma: no cover

if typing.TYPE_CHECKING:
    from keras.api._v2 import keras  # noqa: F811 # pragma: no cover


__all__ = [
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_BETA_1",
    "DEFAULT_BETA_2",
    "QOConstants",
]

DEFAULT_LEARNING_RATE: float = 0.008
DEFAULT_BETA_1: float = 0.999
DEFAULT_BETA_2: float = 0.9999


class Sample(tf.Tensor, Sequence[float]):
    pass


class QOConstants(Model):

    optimizer: keras.optimizers.Optimizer = Field(
        default=keras.optimizers.Adam(
            learning_rate=DEFAULT_LEARNING_RATE,
            beta_1=DEFAULT_BETA_1,
            beta_2=DEFAULT_BETA_2,
        )
    )
    tracker: QOTracker = Field(default_factory=QOTracker)

    k: float = Field(default=4.0)
    mass: float = Field(default=1.0)
    x_left: float = Field(default=-6.0)
    x_right: float = Field(default=6.0)
    fb: float = Field(default=0.0)
    sample_size: int = Field(default=1000)
    # regularization multipliers
    v_f: float = Field(default=1.0)
    v_lambda: float = Field(default=1.0)
    v_drive: float = Field(default=1.0)

    class Config(Model.Config):
        allow_mutation = True
        arbitrary_types_allowed = True

    @cached_property
    def sample(self) -> Sample:
        return cast(
            Sample,
            tf.cast(
                tf.reshape(
                    tf.linspace(self.x_left, self.x_right, self.sample_size),
                    shape=(-1, 1),
                ),
                dtype=tf.float32,
            ),
        )

    def get_sample(self) -> Sample:
        return self.sample
