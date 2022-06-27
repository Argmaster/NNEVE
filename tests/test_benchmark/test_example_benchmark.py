from pathlib import Path
from timeit import timeit
from typing import Any, Callable, Iterable, List, Tuple

import tensorflow as tf

from nneve.utility import disable_gpu_or_skip, plot_multi_sample
from nneve.utility.plotting import pretty_bytes

DIR = Path(__file__).parent
DATA_DIR = DIR / "data"


class TestTensorTensorVsTensorConstant:
    def run_benchmark_samples(
        self,
        function: Callable[..., Any],
        data_set: Iterable[Tuple[Any, ...]],
        samples_in_single: int,
    ) -> List[float]:
        # let it train the graph before to avoid unstable results
        data_set = list(data_set)
        timeit(
            "function(*sample_input)",
            globals={
                "function": function,
                "sample_input": data_set[0],
            },
            number=samples_in_single,
        )
        results: List[float] = []
        for sample_input in data_set:
            results.append(
                timeit(
                    "function(*sample_input)",
                    globals={
                        "function": function,
                        "sample_input": sample_input,
                    },
                    number=samples_in_single,
                )
                / samples_in_single
            )

        return results

    def benchmark_constant_ones(self):
        @tf.function
        def __function(__x: tf.Tensor, __c: tf.Tensor):  # pragma: no cover
            return tf.multiply(__x, __c)

        samples_in_single = 10
        constant = 32.21
        test_range = range(0, 4)

        data_set_1 = []
        data_set_2 = []
        for i in test_range:
            shape = (2 << i,)
            tensor = tf.random.normal(shape, dtype=tf.float64)
            data_set_1.append((tensor, constant))
            data_set_2.append(
                (
                    tensor,
                    tf.constant(constant, shape=shape, dtype=tf.float64),
                )
            )
            del shape

        sample_1 = self.run_benchmark_samples(
            __function,
            data_set=data_set_1,
            samples_in_single=samples_in_single,
        )

        sample_2 = self.run_benchmark_samples(
            __function,
            data_set=data_set_2,
            samples_in_single=samples_in_single,
        )

        plot_multi_sample(
            sample_1,
            sample_2,
            x_range=[pretty_bytes(2 << i) for i in test_range],
            labels=(
                "tensor x constant",
                "tensor x tensor",
            ),
            x_axis_label="Tensor size [float64]",
        )

    def test_constant_ones_cpu(self):
        disable_gpu_or_skip()
        self.benchmark_constant_ones()
