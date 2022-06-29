from typing import Tuple

from pydantic import Field

from nneve.common import Model

__all__ = ["QOParams"]


class QOParams(Model):

    c: float = Field(default=-2.0)
    c_step: float = Field(default=0.16)

    class Config:
        allow_mutation = True
        arbitrary_types_allowed = True

    def update(self) -> None:
        self.c += self.c_step

    def get_extra(self) -> Tuple[float, ...]:
        return (self.c,)
