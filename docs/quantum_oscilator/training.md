There are 3 methods to train a neural network (train_generations, train and
train_step). In train_generations the generation of test trials is defined. The
train function is used to train the neural network on a given number of epochs.
At the end of each test trial, a copy of the model containing the parameters
(e.g., weights) of the trained model is returned . In turn, train_step
implements the individual step of learning the network using the previously
implemented loss function. Train step returns the average loss value of the
neural network for the given problem. You can manipulate the behavior of the
network when training it. Parameters that we can optimize are: generations ->
number of generations of training trials; epochs -> number of epochs of
training the network, plot -> auxiliary parameter for retaining individual
neural network learning values.
