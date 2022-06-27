# `#!python class QOParams(pydantic.BaseModel)`

## Parent classes

[`#!python class pydantic.BaseModel`](https://pydantic-docs.helpmanual.io/usage/models/#basic-model-usage)

## Introduction

QOParams class is responsible for holding variables which can change during the
learning of QONetwork. This class can (should) also change the values of the
variables in using a dedicated method that is called after each generation of
the learning process.

## Instance attributes

!!! note

    - Attributes are mutable
    - Arbitrary types are allowed to be used as attribute values

### `#!python c: float`

While searching for eigenvalues, learning algorithm increases value of c by
`0.16` every genration, which increases Ldrive. That forces the network to
search for larger eigenvalues and the associated eigenfunctions\_

## Instance methods

### `#!python def update(self) -> None`

Called after each network learning generation to (optionally) change the values
of the variables stored in the attributes of the QOParams class

### `#!python def extra(self) -> Tuple[float, ...]`

Returns a tuple containing the variables to be passed to the cost function. The
order of the variables is arbitrary, but note that the tuple is unpacked in the
function call. This method was created to overcome the problem of passing class
instances to tensorflow graph functions.
