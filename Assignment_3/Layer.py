
import numpy

class Layer():
    #input-> number_of_nodes
    def __init__(self, number_nodes, previous_layer, activation_type):
        self.nodes = np.array(number_nodes)
        self.weightMatrix = np.empty([number_nodes, len(previous_layer.nodes)])
        self.activation = activation_type

        

def activation_funtion():
    if self.activation == 'sigmoid':
        pass

    if self.activation == 'linear':
        pass

# def linear_funtion():
#     pass

def weightMatrix(self):
    
