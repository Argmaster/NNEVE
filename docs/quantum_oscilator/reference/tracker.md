# `#!python class QOTracker(pydantic.BaseModel)`

## Parent classes

[`#!python class pydantic.BaseModel`](https://pydantic-docs.helpmanual.io/usage/models/#basic-model-usage)

!!! note

    `#!python class QOTracker` is an element of internal API of
    <span style="color:#cf970a;">**nneve**</span>.<span style="color:#0d75fc;">**quantum_oscilator**</span>
    subpackage was only ment to interact with `#!python class QONetwork` and `#!python class QOConstants`.

`#!python class QOTracker` is responsible for gathering network state metrics
during network learning process. The class is also equipped with methods to
plot graphs based on the collected data. The class is also equipped with
methods to plot graphs based on the collected data. Its additional ability is
to create network status messages that are shown on the command line while the
network is being learned.

## Instance attributes

### `#!python c: float`

### `#!python total_loss: List[float] = Field(default_factory=list)`

Stores history of total_loss values which were presented by the network during
training process.

### `#!python eigenvalue: List[float] = Field(default_factory=list)`

Stores history of eigenvalue values which were presented by the network during
training process.

### `#!python residuum: List[float] = Field(default_factory=list)`

Stores history of residuum values which were presented by the network during
training process.

### `#!python function_loss: List[float] = Field(default_factory=list)`

Stores history of function_loss values which were presented by the network
during training process.

### `#!python eigenvalue_loss: List[float] = Field(default_factory=list)`

Stores history of eigenvalue_loss values which were presented by the network
during training process.

### `#!python drive_loss: List[float] = Field(default_factory=list)`

Stores history of drive_loss values which were presented by the network during
training process.

### `#!python c: List[float] = Field(default_factory=list)`

Stores history of c values which were presented by the network during training
process.

## Instance methods

### `#!python def push_stats(self, ...) -> None`

#### Parameters

Appends passed positional arguments to corresponding histories.

| name            | type             | description                                           |
| --------------- | ---------------- | ----------------------------------------------------- |
| total_loss      | `#!python float` | value to be appended to total_loss history list.      |
| eigenvalue      | `#!python float` | value to be appended to eigenvalue history list.      |
| residuum        | `#!python float` | value to be appended to residuum history list.        |
| function_loss   | `#!python float` | value to be appended to function_loss history list.   |
| eigenvalue_loss | `#!python float` | value to be appended to eigenvalue_loss history list. |
| drive_loss      | `#!python float` | value to be appended to drive_loss history list.      |
| c               | `#!python float` | value to be appended to c history list.               |
| \*\_            | `#!python Any`   | extra arguments are ignored.                          |

#### Returns:

| type                | description          |
| ------------------- | -------------------- |
| `#!python NoneType` | no value is returned |
