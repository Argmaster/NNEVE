import logging
import pickle
import typing
from contextlib import suppress
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Iterable, List, Tuple, cast

import tensorflow as tf
from matplotlib import pyplot as plt
from rich.progress import Progress
from tensorflow import keras

from .constants import QOConstants
from .layers import Eigenvalue
from .params import QOParams

if typing.TYPE_CHECKING:
    from keras.api._v2 import keras  # noqa: F811

LossFunctionT = Callable[
    [tf.Tensor, tf.Variable, float],
    Tuple[tf.Tensor, Tuple[Any, ...]],
]


class QONetwork(keras.Model):

    constants: QOConstants
    is_debug: bool
    loss_function: LossFunctionT
    is_console_mode: bool

    def __init__(
        self,
        constants: QOConstants,
        is_debug: bool = False,
        is_console_mode: bool = True,
        name: str = "QONetwork",
    ):
        self.constants = constants
        self.is_console_mode = is_console_mode
        self.is_debug = is_debug
        inputs, outputs = self.assemble_hook()
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            name=name,
        )
        self.loss_function = self.get_loss_function()
        self.compile(
            self.constants.optimizer,
            jit_compile=True,
        )
        assert self.name is not None

    def assemble_hook(
        self,
    ) -> Tuple[List[keras.layers.InputLayer], List[keras.layers.Dense]]:
        # 1 value input layer
        inputs = cast(
            keras.layers.InputLayer,
            keras.Input(
                shape=(1,),
                name="input",
                dtype=tf.float32,
            ),
        )
        # One neuron decides what value should λ have
        eigenvalue_out = Eigenvalue(name="eigenvalue")(inputs)
        input_and_eigenvalue = keras.layers.Concatenate(
            axis=1,
            name="join",
        )([inputs, eigenvalue_out])
        # two dense layers, each with {neurons} neurons as NN body
        d1 = keras.layers.Dense(
            2,
            activation=tf.sin,
            name="dense_1",
            dtype=tf.float32,
        )(input_and_eigenvalue)
        # internal second layer
        d2 = keras.layers.Dense(
            self.constants.neuron_count,
            activation=tf.sin,
            name="dense_2",
            dtype=tf.float32,
        )(d1)
        d3 = keras.layers.Dense(
            self.constants.neuron_count,
            activation=tf.sin,
            name="dense_3",
            dtype=tf.float32,
        )(d2)
        # single value output from neural network
        outputs = keras.layers.Dense(
            1,
            # ; activation=tf.sin,
            name="predictions",
            dtype=tf.float32,
        )(d3)
        # single output from full network, λ is accessed by single call
        # to "eigenvalue" Dense layer - much cheaper op
        return [inputs], [outputs, eigenvalue_out]

    def get_loss_function(self) -> LossFunctionT:  # noqa: CFQ004, CFQ001
        @tf.function
        def loss_function(
            x: tf.Tensor,
            deriv_x: tf.Variable,
            c: tf.Tensor,
        ) -> Tuple[tf.Tensor, Tuple[Any, ...]]:  # pragma: no cover

            current_eigenvalue = self(x)[1][0]

            with tf.GradientTape() as second:
                with tf.GradientTape() as first:
                    psi, _ = parametric_solution(deriv_x)  # type: ignore
                dy_dx = first.gradient(psi, deriv_x)
            dy_dxx = second.gradient(dy_dx, deriv_x)

            residuum = tf.square(
                tf.divide(dy_dxx, -2.0 * self.constants.mass)
                + (potential(x) * psi)
                - (current_eigenvalue * psi)
            )
            function_loss = tf.divide(
                1,
                tf.add(tf.reduce_mean(tf.square(psi)), 1e-6),
            )
            lambda_loss = tf.divide(
                1, tf.add(tf.reduce_mean(tf.square(current_eigenvalue)), 1e-6)
            )
            drive_loss = tf.exp(
                tf.reduce_mean(tf.subtract(c, current_eigenvalue))
            )
            total_loss = residuum + function_loss + lambda_loss + drive_loss
            return total_loss, (
                tf.reduce_mean(total_loss),
                current_eigenvalue,
                tf.reduce_mean(residuum),
                function_loss,
                lambda_loss,
                drive_loss,
                c,
                0.0,
            )  # type: ignore

        @tf.function
        def parametric_solution(
            x: tf.Variable,
        ) -> Tuple[tf.Tensor, tf.Tensor]:  # pragma: no cover
            psi, current_eigenvalue = self(x)
            return (
                tf.add(
                    tf.constant(self.constants.fb, dtype=tf.float32),
                    tf.multiply(boundary(x), psi),
                ),
                current_eigenvalue[0],
            )

        @tf.function
        def boundary(x: tf.Tensor) -> tf.Tensor:  # pragma: no cover
            return (1 - tf.exp(tf.subtract(self.constants.x_left, x))) * (
                1 - tf.exp(tf.subtract(x, self.constants.x_right))
            )

        @tf.function
        def potential(x: tf.Tensor) -> tf.Tensor:  # pragma: no cover
            return tf.divide(tf.multiply(self.constants.k, tf.square(x)), 2)

        if self.is_debug:  # pragma: no cover
            self._potential_function = potential
            self._boundary_function = boundary
            self._parametric_solution_function = parametric_solution
            self._loss_function_function = loss_function

        self.parametric_solution = parametric_solution

        return cast(LossFunctionT, loss_function)

    def train_generations(
        self,
        params: QOParams,
        generations: int = 5,
        epochs: int = 1000,
    ) -> Iterable["QONetwork"]:
        with suppress(KeyboardInterrupt):
            for i in range(generations):
                logging.info(
                    f"Generation: {i + 1:4.0f} our of "
                    f"{generations:.0f}, ({i / generations:.2%})"
                )
                self.train(params, epochs)
                params.update()
                yield self

    def train(  # noqa: CCR001
        self, params: QOParams, epochs: int = 10
    ) -> None:
        x = self.constants.sample()
        if self.is_console_mode:
            with Progress() as progress:
                task = progress.add_task(
                    description="Learning...", total=epochs + 1
                )

                for i in range(epochs):
                    self._train_step(x, params)

                    description = self.constants.tracker.get_trace(i)

                    progress.update(
                        task,
                        advance=1,
                        description=description.capitalize(),
                    )
        else:
            for i in range(epochs):
                self._train_step(x, params)
                description = self.constants.tracker.get_trace(i)
                logging.info(description)

    def _train_step(self, x: tf.Tensor, params: QOParams) -> float:
        deriv_x = tf.Variable(initial_value=x)

        with tf.GradientTape() as tape:
            loss_value, stats = self.loss_function(
                x,
                deriv_x,
                *params.extra(),
            )

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss_value, trainable_vars)
        self.constants.optimizer.apply_gradients(
            zip(gradients, trainable_vars)
        )
        # to make this push loss function agnostic
        average_loss = tf.reduce_mean(loss_value)
        self.constants.tracker.push_stats(*stats)

        return average_loss

    def get_deepcopy(self) -> "QONetwork":
        constants_copy = self.constants.copy(deep=True)
        weights_copy = deepcopy(self.get_weights())
        model_copy = self.__class__(constants=constants_copy)
        model_copy.set_weights(weights_copy)
        return model_copy

    def save(self, filepath: Path) -> None:  # noqa: FNE003
        weights = self.get_weights()
        with filepath.open("wb") as file:
            pickle.dump(weights, file)

    def load(self, filepath: Path) -> None:  # noqa: FNE004
        with filepath.open("rb") as file:
            weights = pickle.load(file)
        self.set_weights(weights)

    def plot_solution(self) -> None:
        x = self.constants.sample()
        plt.plot(x, self(x)[0])
