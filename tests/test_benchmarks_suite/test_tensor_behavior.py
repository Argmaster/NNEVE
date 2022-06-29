from pathlib import Path
from timeit import timeit
from typing import Any, Callable, Iterable, List, Tuple

import pytest
import tensorflow as tf
from matplotlib import pyplot as plt

from nneve.utility import (
    datetag,
    disable_gpu_or_skip,
    get_sys_info,
    is_gpu,
    plot_multi_sample,
    skip_if_no_gpu,
)
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

    def dump_plot(self, plot_title: str):
        plt.title(f"{plot_title}".capitalize())
        plot_title_no_spaces = plot_title.replace(" ", "_")
        dest_dir = DATA_DIR / plot_title_no_spaces
        dest_dir.mkdir(0o777, True, True)
        dtag = datetag()
        plt.savefig(dest_dir / f"{plot_title_no_spaces}_{dtag}.png")
        (dest_dir / f"sys_info_{plot_title_no_spaces}_{dtag}.json").write_text(
            get_sys_info().json(), encoding="utf-8"
        )

    def benchmark_constant_ones(self):
        @tf.function
        def __function(__x: tf.Tensor, __c: tf.Tensor):
            return tf.multiply(__x, __c)

        samples_in_single = 100
        constant = 32.21
        if is_gpu():
            test_range = range(0, 26)
        else:
            test_range = range(0, 26)

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

    @pytest.mark.benchmark()
    def test_constant_ones_cpu(self):
        disable_gpu_or_skip()
        self.benchmark_constant_ones()
        self.dump_plot("constant ones cpu")

    @pytest.mark.benchmark()
    def test_constant_ones_gpu(self):
        skip_if_no_gpu()
        self.benchmark_constant_ones()
        self.dump_plot("constant ones gpu")

    def benchmark_graph_non_graph(self):

        samples_in_single = 40
        test_range = range(0, 20)

        def __function(__x: tf.Tensor, __c: tf.Tensor, __i: int):
            total = tf.zeros_like(__x)
            for _ in range(__i):
                total += tf.multiply(__c, __x)
            return total

        data_set = []
        for i in test_range:
            shape = (2 << i,)
            tensor1 = tf.random.normal(shape, dtype=tf.float64)
            tensor2 = tf.random.normal(shape, dtype=tf.float64)
            data_set.append((tensor1, tensor2, 40))

        sample_1 = self.run_benchmark_samples(
            __function,
            data_set=data_set,
            samples_in_single=samples_in_single,
        )

        sample_2 = self.run_benchmark_samples(
            tf.function(__function),
            data_set=data_set,
            samples_in_single=samples_in_single,
        )

        plot_multi_sample(
            sample_1,
            sample_2,
            x_range=[pretty_bytes(2 << i) for i in test_range],
            labels=(
                "non graph",
                "graph",
            ),
            x_axis_label="Tensor size [float64]",
        )

    @pytest.mark.benchmark()
    def test_graph_non_graph_cpu(self):
        disable_gpu_or_skip()
        self.benchmark_graph_non_graph()
        self.dump_plot("graph non graph cpu")

    @pytest.mark.benchmark()
    def test_graph_non_graph_gpu(self):
        skip_if_no_gpu()
        self.benchmark_graph_non_graph()
        self.dump_plot("graph non graph gpu")
