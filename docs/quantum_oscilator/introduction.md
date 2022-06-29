# Introduction for Quantum Oscilator problem

The goal of this project was not so much to implement the network itself, but
to verify how efficiently differential equations with an eigenproblem can be
solved using neural networks. Unfortunately, as a result of this experiment, we
came to the conclusion that the process of learning network is too
time-consuming and in this form can not be applied to other problems that we
considered as a target.

As sad as it sounds, we leave the implementation of the network for all
interested parties to see. On the other hand, we ourselves are changing the
space of our interest from equations with an eigenproblem to equations without
one.

## Maths behind the problem

Consider a mathematical problem of the following form

<figure markdown>
  $$
  \mathcal{L}f(x) = \lambda f(x)
  $$
  <figcaption>(1)</figcaption>
</figure>

- $x$ a variable
- $\mathcal{L}$ is a differential operator which depends on $x$ and its
  derivatives
- $f(x)$ is the eigenfunction
- $\lambda$ is the eigenvalue associated with eigenfunction

##### Network input-output

Our neural network in form of represented by $N(x, \lambda )$ is expected to
get two values as input:

- $x$ - single floating point value within $x_L$ to $x_R$ range
- $\lambda$ - eigenvalue approximation selected by neural network

As output, we also expect two values:

- $y$ - floating point value corresponding to single $x$ value within $x_L$ to
  $x_R$ range - $f(x)$ approximation
- $\lambda$ - eigenvalue approximation selected by neural network

To use our neural network as a replacement for $f(x)$ we need to put the values
returned from it into the equation below

<figure markdown>
  $$
  f(x,λ) = f_b + g(x)N(x,λ)
  $$
  <figcaption>(2)</figcaption>
</figure>

- $f_b$ arbitrary constant
- $g(x)$ [boundary condition](/quantum_oscilator/introduction/#equation-3)
- $N(x, \lambda )$ - our neural network

<figure markdown>
  $$
  g(x) = (1 −e^{−(x−x_L)})(1 − e^{−(x−x_R)})
  $$
  <figcaption>(3)</figcaption>
</figure>

- $x_L$ minimum (left) value in the examined range
- $x_R$ maximum (right) value in the examined range

## Loss function

<figure markdown>
  $$
  \mathcal{L}f(x) - λf(x) = 0
  $$
  <figcaption>(4)</figcaption>
</figure>

By rearranging [(1)](/quantum_oscilator/introduction/#maths-behind-the-problem)
to form (4) we are able to use it to measure how far from exact eigenfunction
and exact eigenvalue is current state of neural network. To achieve this we
have to replace $f(x)$ with $f(x, \lambda )$ and use $\lambda$ retuned from
network.

Therefore without any prepared input data, just based on loss function and back
propagation we can move from a random state of network to an approximation of
the equation solution.

## Regularizators

To find multiple solutions of given equation with this neural network, it is
necessary to add additional components to the loss function which will prevent
the neural network from staying in trivial solution state and force transitions
between successive states.

<figure markdown>
  $$
  L_f = \frac{1}{f(x, \lambda )^2}
  $$
  <figcaption>(5) - avoid trivial solutions</figcaption>
</figure>

<figure markdown>
  $$
  L_{\lambda} = \frac{1}{\lambda^2}
  $$
  <figcaption>(6) - avoid trivial solutions</figcaption>
</figure>

<figure markdown>
  $$
  L_{drive} = e ^ {-\lambda + c}
  $$
  <figcaption>(7) - push to subsequent solutions</figcaption>
</figure>

- $c$ - variable increased in regular intervals

## Network structure

The network is a simple sequential model. The only unusual thing is returning a
value from the lambda layer which is also the input to the network. This value
is necessary to calculate the value of the loss function, and getting it in any
other clever way breaks the whole learning process.

```mermaid
%%{ init : { "flowchart" : { "curve" : "basisOpen" }}}%%
flowchart LR
    subgraph Input
        X((x))
        b1((1)) ---> λ((λ))
    end
    subgraph Deep 0
        n00((neuron 1))
        n01((neuron 2))
        n02((neuron ...))
        n03((neuron 50))

        X ---> n00
        X ---> n01
        X ---> n02
        X ---> n03

        λ ---> n00
        λ ---> n01
        λ ---> n02
        λ ---> n03
    end
    subgraph Deep 1
        n10((neuron 1))
        n11((neuron 2))
        n12((neuron ...))
        n13((neuron 50))

        n00 ---> n10
        n00 ---> n11
        n00 ---> n12
        n00 ---> n13

        n01 ---> n10
        n01 ---> n11
        n01 ---> n12
        n01 ---> n13

        n02 ---> n10
        n02 ---> n11
        n02 ---> n12
        n02 ---> n13

        n03 ---> n10
        n03 ---> n11
        n03 ---> n12
        n03 ---> n13
    end
    subgraph Output
        N(("N(x, λ)"))
        n10 ---> N
        n11 ---> N
        n12 ---> N
        n13 ---> N
    end
```

## Code flow

```mermaid
flowchart LR
QOT((QOTracker))
QOC((QOConstants))
QON((QONetwork))
QOP((QOParams))
tr1["train_generations(...)"]
tr2["train(...)"]
  QOT --> QOC --> QON --> tr1
  QOP --> tr1
  tr1 ---> tr2 ---> tr1
```
