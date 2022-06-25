# *QOConstants* class
## Purpose
QOConstants class contains constants and configuration of our neural network model. It inherits after the pydantics's BaseModel class to ensure compatibility of instances of our neural network with defined model.
## Constants
* *optimizer*: argument required for compiling a Keras model
* *tracker*: QOTracker class, responsible for collecting data of our neural network learning process
* *k*: force constant
* *mass*: Planck mass
* *x_left*: left boundary condition of our quantum harmonic oscillator model
* *x_right*: right boundary condition of our quantum harmonic oscillator model
* *fb*: constant boundary value for boundary conditions
* *sample_size*: size of our  current learning sample (number of points on the linear space)
* *neuron_count*: defines number of neurons 
* *v_f*: regularization function which prevents our network from learning trivial eigenfunctions
* *v_lambda*: regularization function which prevents our network from learning trivial eigenvalues
* *v_drive*: regularization function which motivates our network to scan for higher values of eigenvalues
* *__sample*: current learning sample
## *Config* class
*Config* class contains and defines parameters of our neural network model.