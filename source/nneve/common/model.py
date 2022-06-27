import sys

from pydantic import BaseModel

if sys.version_info < (3, 8):
    from backports.cached_property import (  # noqa # pragma: no cover
        cached_property,
    )
else:
    from functools import cached_property  # noqa # pragma: no cover


__all__ = ["Model"]


class Model(BaseModel):
    class Config:
        keep_untouched = (cached_property,)
