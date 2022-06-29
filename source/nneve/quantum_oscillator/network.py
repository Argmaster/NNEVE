import logging
import pickle
import sys
import typing
from contextlib import suppress
from pathlib import Path
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    cast,
)

import tensorflow as tf
from matplotlib import pyplot as plt
from rich.progress import Progress
from tensorflow import keras

from .constants import QOConstants
from .layers import Eigenvalue
from .params import QOParams

if sys.version_info < (3, 8):
    from backports.cached_property import (  # noqa # pragma: no cover
        cached_property,
    )
else:
    from functools import cached_property  # noqa # pragma: no cover

if typing.TYPE_CHECKING:
    from keras.api._v2 import keras  # noqa: F811

LossFunctionT = Callable[
    [tf.Tensor, float],
    Tuple[tf.Tensor, Tuple[Any, ...]],
]


class QONetwork(keras.Model):

    constants: QOConstants
    is_debug: bool
    is_console_mode: bool

    def __init__(
        self,
        constants: Optional[QOConstants] = None,
        is_debug: bool = False,
        is_console_mode: bool = True,
        name: str = "QONetwork",
    ):
        self.constants = constants if constants is not None else QOConstants()
        self.is_console_mode = is_console_mode
        self.is_debug = is_debug
        # attribute access to cache function value
        self.loss_function
        # create network structure with two 50 neuron deep layers
        inputs, outputs = self.assemble_hook((50, 50))
        # with constructor of keras.Model initialize model machinery
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            name=name,
        )
        # not really needed thanks to custom train loop
        # but is called to satisfy internals of keras
        self.compile(
            self.constants.optimizer,
            jit_compile=True,
        )
        assert self.name is not None

    def assemble_hook(
        self, deep_layers: Sequence[int] = (50, 50)
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
        # dense layer seem not to be willing to accept two inputs
        # at the same time, therefore λ and x are bound together
        # with help of another dense layer
        d1 = keras.layers.Dense(
            2,
            activation=tf.sin,
            name="dense_input",
            dtype=tf.float32,
        )(input_and_eigenvalue)
        # dynamically defined number of deep layers, specified by
        # input argument `deep_layers`. They are joined sequentially
        deep_in = d1
        deep_out = None
        for index, neuron_count in enumerate(deep_layers):
            deep_out = keras.layers.Dense(
                neuron_count,
                activation=tf.sin,
                name=f"dense_{index}",
                dtype=tf.float32,
            )(deep_in)
            deep_in = deep_out
            del index  # make sure no references by mistake
        # We require at least one deep layer to be available, otherwise
        # this network makes no sense
        assert deep_out is not None
        del deep_in  # make sure no references by mistake
        # single value output from neural network
        outputs = keras.layers.Dense(
            1,
            name="predictions",
            dtype=tf.float32,
        )(deep_out)
        # eigenvalue is not acquired by single layer call because it
        # was breaking learning process
        return [inputs], [outputs, eigenvalue_out]

    @cached_property
    def loss_function(self) -> LossFunctionT:  # noqa: CFQ004, CFQ001
        @tf.function  # type: ignore
        def loss_function_impl(
            x: tf.Tensor,
            c: tf.Tensor,
        ) -> Tuple[tf.Tensor, Tuple[Any, ...]]:  # pragma: no cover
            # for more extensive description of loss function visit
            # https://argmaster.github.io/NNEVE/quantum_oscillator/introduction/
            current_eigenvalue = self(x)[1][0]
            deriv_x = tf.identity(self.constants.get_sample())

            with tf.GradientTape() as second:
                second.watch(deriv_x)

                with tf.GradientTape() as first:
                    first.watch(deriv_x)
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
            )  # type: ignore

        @tf.function  # type: ignore
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

        @tf.function  # type: ignore
        def boundary(x: tf.Tensor) -> tf.Tensor:  # pragma: no cover
            return (1 - tf.exp(tf.subtract(self.constants.x_left, x))) * (
                1 - tf.exp(tf.subtract(x, self.constants.x_right))
            )

        @tf.function  # type: ignore
        def potential(x: tf.Tensor) -> tf.Tensor:  # pragma: no cover
            return tf.divide(tf.multiply(self.constants.k, tf.square(x)), 2)

        if self.is_debug:
            self.debug_potential_function = potential
            self.debug_boundary_function = boundary
            self.debug_parametric_solution_function = parametric_solution
            self.debug_loss_function_function = loss_function_impl

        self.parametric_solution = parametric_solution

        return cast(LossFunctionT, loss_function_impl)

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
        x = self.constants.get_sample()
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

        with tf.GradientTape() as tape:
            loss_value, stats = self.loss_function(x, *params.get_extra())

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss_value, trainable_vars)
        self.constants.optimizer.apply_gradients(
            zip(gradients, trainable_vars)
        )
        # to make this push loss function agnostic
        average_loss = tf.reduce_mean(loss_value)
        self.constants.tracker.push_stats(*stats)

        return float(average_loss)

    def save(self, filepath: Path) -> None:  # noqa: FNE003
        weights = self.get_weights()
        with filepath.open("wb") as file:
            pickle.dump(weights, file)

    def load(self, filepath: Path) -> None:  # noqa: FNE004
        with filepath.open("rb") as file:
            weights = pickle.load(file)
        self.set_weights(weights)

    def plot_solution(self) -> None:
        x = self.constants.get_sample()
        plt.plot(x, self(x)[0])
