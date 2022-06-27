from typing import List, Optional, Sequence

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import BaseModel, Field


class QOTracker(BaseModel):

    eigenvalue: List[float] = Field(default_factory=list)
    residuum: List[float] = Field(default_factory=list)
    drive_loss: List[float] = Field(default_factory=list)
    function_loss: List[float] = Field(default_factory=list)
    eigenvalue_loss: List[float] = Field(default_factory=list)
    c: List[float] = Field(default_factory=list)

    total_loss: List[float] = Field(default_factory=list)

    def push_stats(  # noqa: CFQ002
        self,
        total_loss: float,
        eigenvalue: float,
        residuum: float,
        function_loss: float,
        eigenvalue_loss: float,
        drive_loss: float,
        c: float,
        *_: float,
    ) -> None:
        self.total_loss.append(float(total_loss))
        self.eigenvalue.append(float(eigenvalue))
        self.residuum.append(float(residuum))
        self.function_loss.append(float(function_loss))
        self.eigenvalue_loss.append(float(eigenvalue_loss))
        self.drive_loss.append(float(drive_loss))
        self.c.append(c)

    def get_trace(self, index: int) -> str:
        return (
            f"epoch: {index:<6.0f} loss: {self.total_loss[-1]:<10.4f} λ: "
            f"{self.eigenvalue[-1]:<10.4f} c: {self.c[-1]:<5.2f}"
        )

    def plot(
        self,
        solution_y: Sequence[float],
        solution_x: Sequence[float],
    ) -> Figure:
        fig, (
            (
                eigenvalue_ax,
                residuum_ax,
                function_loss_ax,
                eigenvalue_loss_ax,
                total_loss_ax,
            ),
            (
                drive_loss_ax,
                c_ax,
                _,
                all_ax,
                solution_ax,
            ),
        ) = plt.subplots(
            2, 5, figsize=(20, 8), dpi=300, constrained_layout=True
        )  # type: ignore
        # Completely hides an unused blank plot
        _.axis("off")
        self._plot(self.eigenvalue, where=eigenvalue_ax, label="Eigenvalue")
        self._plot(
            self.residuum,
            where=residuum_ax,
            label="Residuum",
            logy=True,
        )
        self._plot(
            self.function_loss,
            where=function_loss_ax,
            label="Function loss",
            logy=True,
        )
        self._plot(
            self.eigenvalue_loss,
            where=eigenvalue_loss_ax,
            label="Eigenvalue loss",
            logy=True,
        )
        self._plot(
            self.total_loss,
            where=total_loss_ax,
            label="Total Loss (R+Lf+Le+Ld)",
            logy=True,
        )
        self._plot(
            self.drive_loss,
            where=drive_loss_ax,
            label="Drive loss",
            logy=True,
        )
        self._plot(
            self.c,
            where=c_ax,
            label="Drive (C)",
        )
        self._plot(
            self.residuum,
            self.total_loss,
            self.function_loss,
            self.eigenvalue_loss,
            self.drive_loss,
            self.c,
            where=all_ax,
            label="All",
            logy=True,
        )
        self._plot(
            solution_y,
            where=solution_ax,
            label="Solution",
            x=solution_x,
        )
        return fig

    def _plot(  # noqa: CFQ002, CCR001
        self,
        *what: Sequence[float],
        where: Axes,
        label: str = "",
        logy: bool = False,
        logx: bool = False,
        x: Optional[Sequence[float]] = None,
    ) -> None:
        for sequence in what:
            if x is None:
                _x = np.arange(len(sequence))
            else:
                _x = x

            if logy and not logx:
                where.semilogy(_x, sequence)
            elif not logy and logx:
                where.semilogx(_x, sequence)
            elif logy and logx:
                where.loglog(_x, sequence)
            else:
                where.plot(_x, sequence)
            where.set_title(label)
        where.grid()
