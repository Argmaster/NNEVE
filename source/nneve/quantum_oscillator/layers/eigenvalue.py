import typing
from typing import Optional

import tensorflow as tf
from tensorflow import keras

if typing.TYPE_CHECKING:
    from keras.api._v2 import keras  # noqa: F811


class Eigenvalue(keras.layers.Layer):

    w: tf.Variable

    def __init__(self, name: str, dtype: tf.DType = tf.float32):
        super().__init__(name=name)
        self.w: tf.Variable = self.add_weight(
            dtype=dtype,
            shape=(1, 1),
            initializer=keras.initializers.constant(0.0),
            trainable=True,
        )
        self.b: tf.Variable = self.add_weight(
            dtype=dtype,
            shape=(1, 1),
            initializer=keras.initializers.constant(0.0),
            trainable=True,
        )

    def call(self, inputs: Optional[tf.Tensor]) -> tf.Tensor:
        eigenvalue = tf.add(
            tf.multiply(tf.ones_like(inputs, dtype=self.dtype), self.w), self.b
        )
        return eigenvalue
