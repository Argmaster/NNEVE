import typing
from typing import Optional, Sequence, cast

import tensorflow as tf
from pydantic import BaseModel, Field
from tensorflow import keras

from .tracker import QOTracker

if typing.TYPE_CHECKING:
    from keras.api._v2 import keras  # noqa: F811
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


class QOConstants(BaseModel):

    optimizer: keras.optimizers.Optimizer = Field(
        default=keras.optimizers.Adam(
            learning_rate=DEFAULT_LEARNING_RATE,
            beta_1=DEFAULT_BETA_1,
            beta_2=DEFAULT_BETA_2,
        )
    )
    tracker: QOTracker = Field(default_factory=QOTracker)

    k: float
    mass: float
    x_left: float
    x_right: float
    fb: float
    sample_size: int = Field(default=500)
    # network configuration
    neuron_count: int = Field(default=50)
    # regularization multipliers
    v_f: float = Field(default=1.0)
    v_lambda: float = Field(default=1.0)
    v_drive: float = Field(default=1.0)
    __sample: Optional[tf.Tensor] = None

    class Config:
        allow_mutation = True
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True

    def sample(self) -> Sample:
        if self.__sample is None:
            self.__sample = tf.cast(
                tf.reshape(
                    tf.linspace(self.x_left, self.x_right, self.sample_size),
                    shape=(-1, 1),
                ),
                dtype=tf.float32,
            )
        return cast(Sample, self.__sample)
