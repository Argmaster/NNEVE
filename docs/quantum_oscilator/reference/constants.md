# `#!python class QOConstants(pydantic.BaseModel)`

## Parent classes

[`#!python class pydantic.BaseModel`](https://pydantic-docs.helpmanual.io/usage/models/#basic-model-usage)

## Introduction

`#!python class QOConstants` contains physical constants used in loss function
during neural network learning process. It inherits from the pydantic's
BaseModel class to guarantee field type compatibility and their correct filling
without manual implementation of all checks.

## Instance attributes

!!! note

    - Attributes are mutable
    - Arbitrary types are allowed to be used as attribute values

### `#!python optimizer: keras.optimizers.Optimizer`

Argument required for compiling a Keras model.

### `#!python tracker: nneve.QOTracker`

QOTracker class, responsible for collecting metrics during neural network
learning process.

### `#!python k: float`

Oscillator force constant.

### `#!python mass: float`

Planck mass.

### `#!python x_left: float`

Left boundary condition of our quantum harmonic oscillator model.

### `#!python x_right: float`

Right boundary condition of our quantum harmonic oscillator model.

### `#!python fb: float`

Constant boundary value for boundary conditions.

### `#!python sample_size: int = 500`

Size of our current learning sample (number of points on the linear space).

### `#!python sample_size: int = 50`

Defines number of neurons in internal dense layers responsible for learning
function shape.

### `#!python v_f: int = 1.0`

Multiplier of regularization function which prevents our network from learning
trivial eigenfunctions.

### `#!python v_lambda: int = 1.0`

Multiplier of regularization function which prevents our network from learning
trivial eigenvalues.

### `#!python v_drive: int = 1.0`

Multiplier of regularization function which motivates our network to scan for
higher values of eigenvalues.

### `#!python def sample(self) -> tf.Tensor`

Generates tensor of `sample_size` `float32` values in range from `x_left` to
`x_right` for network learning process.

#### Returns

| type      | description                                  |
| --------- | -------------------------------------------- |
| tf.Tensor | `float32` tensor in shape `(sample_size, 1)` |
