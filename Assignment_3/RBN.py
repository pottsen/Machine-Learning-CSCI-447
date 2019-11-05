import numpy as np

"""
Implement a radial basis function neural network with an arbitrary number of inputs, an arbitrary
number of Gaussian basis functions, and an arbitrary number of outputs. As with the feedforward
network, your program should accept the number of inputs, Gaussians, and outputs. It is your choice
which output activation function is used, but it should be appropriate to the data set being trained.
"""

class RBN():
    #input-> #data_as_2dList, possible_outputs, number_of_hidden_nodes
    def __init__(self, data, output, number_of_nodes, gaussian_funtion_type):
    
