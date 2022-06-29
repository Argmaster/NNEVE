# `#!python class QOConstants(pydantic.BaseModel)`

## Parent classes

[`#!python class pydantic.BaseModel`](https://pydantic-docs.helpmanual.io/usage/models/#basic-model-usage)

## Introduction

`#!python class QOConstants` contains physical constants used in loss function
during neural network learning process. It inherits from the pydantic's
BaseModel class to guarantee field type compatibility and their correct filling
without manual implementation of all checks.

```python title="Example construction with manual values set"
{% include 'examples/code/new_constants.py' %}
```

## Instance attributes

!!! note

    - Attributes are mutable
    - Arbitrary types are allowed to be used as attribute values

### `#!python optimizer: keras.optimizers.Optimizer`

```python
Field(
        default=keras.optimizers.Adam(
            learning_rate=DEFAULT_LEARNING_RATE,
            beta_1=DEFAULT_BETA_1,
            beta_2=DEFAULT_BETA_2,
        )
    )
```

Argument required for compiling a Keras model.

### `#!python tracker: QOTracker`

```python
Field(default_factory=QOTracker)
```

QOTracker class, responsible for collecting metrics during neural network
learning process.

### `#!python k: float`

```python
Field(default=4.0)
```

Oscillator force constant.

### `#!python mass: float`

```python
Field(default=1.0)
```

Oscillator mass used in
$[\frac{\hbar^2}{2m}\frac{\partial^2}{\partial x^2} + V(x)]\psi(x) = E\psi(x)$

### `#!python x_left: float`

```python
Field(default=-6.0)
```

Left boundary condition of our quantum harmonic oscillator model.

### `#!python x_right: float`

```python
Field(default=6.0)
```

Right boundary condition of our quantum harmonic oscillator model.

### `#!python fb: float`

```python
Field(default=0.0)
```

Constant boundary value for boundary conditions.

### `#!python sample_size: int`

```python
Field(default=1000)
```

Size of our current learning sample (number of points on the linear space).

### `#!python v_f: int`

```python
Field(default=1.0)
```

Multiplier of regularization function which prevents our network from learning
trivial eigenfunctions.

### `#!python v_lambda: int`

```python
Field(default=1.0)
```

Multiplier of regularization function which prevents our network from learning
trivial eigenvalues.

### `#!python v_drive: int`

```python
Field(default=1.0)
```

Multiplier of regularization function which motivates our network to scan for
higher values of eigenvalues.

## Instance methods

### `#!python def sample(self) -> tf.Tensor`

Generates tensor of `sample_size` `float32` values in range from `x_left` to
`x_right` for network learning process.

#### Returns

| type      | description                                  |
| --------- | -------------------------------------------- |
| tf.Tensor | `float32` tensor in shape `(sample_size, 1)` |
