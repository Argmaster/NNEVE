from typing import Tuple

from pydantic import BaseModel, Field


class QOParams(BaseModel):

    c: float
    c_step: float = Field(default=0.16)

    class Config:
        allow_mutation = True
        arbitrary_types_allowed = True

    def update(self) -> None:
        self.c += self.c_step

    def extra(self) -> Tuple[float, ...]:
        return (self.c,)
