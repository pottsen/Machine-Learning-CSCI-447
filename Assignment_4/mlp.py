from layer import Layer
import numpy as np

class MLP():

    #includes input and output node sizes - ie:[4,1] is a 0 hidden layer mlp (regression)
    def __init__(self, nodes_per_layer):

        self.layers = []

        for i in range(len(nodes_per_layer)):
            if (i==0):
                prev = 'start'
            else:
                prev = nodes_per_layer[i-1]

            if (i==len(nodes_per_layer) - 1):
                next = 'end'
            else:
                next = nodes_per_layer[i+1]

            self.layers.append(Layer(nodes_per_layer[i],next_layer_number = next, previous_layer_number = prev))

    def __str__(self):
        layer_str = []

        max_nodes = 0

        for i in self.layers:
            layer = str(i)
            layer_str.append(layer.split('\n'))
            if layer.count('\n') > max_nodes:
                max_nodes = layer.count('\n')

        string = ""
        for i in range(max_nodes):
            for j in layer_str:
                if(len(j) > i+1):
                    string += j[i].center(9)
                else:
                    string += "         "
            string += "\n"

        return string

    def print_weights(self):
        iter = 0
        for layer in self.layers:
            print("---matrix "+ str(iter) +"---")
            print(layer.next_weights)
            iter += 1

    def predict(self, point, regression='default'):
        dat = np.transpose([point])
        self.layers[0].set_nodes(dat)
        for i in range(len(self.layers)-1):
            nxt_lyr = self.layers[i].feed_forward_sigmoid()
            self.layers[i+1].set_nodes(nxt_lyr)
        if(regression == 'default'):
            if(self.layers[-1].get_layer_size() == 1):
                regression = True
            else:
                regression = False

        if(regression):
            nxt_lyr = self.layers[-2].feed_forward()
        else:
            nxt_lyr = self.layers[-2].feed_forward_sigmoid()
        self.layers[-1].set_nodes(nxt_lyr)

        #print(self.layers[-1])
        oned = []
        for i in self.layers[-1].nodes:
            oned.append(i[0])
        #print(self.layers[-1])
        return(oned)

    #this method will return a vector representation of the hidden weight matricies for easy cross breeding
    def unzip_neuron(self):
        neuron = []
        for layer in self.layers:
            weights = layer.next_weights
            if(type(weights) != str and weights != 'end'):
                for i in weights:
                    for j in i:
                        neuron.append(j)
        return neuron

    #this method will take a vector representation of hidden weight matricies (neurons_as_vector) and set this MLP's weights to a zipped version of it
    def rezip_neuron(self, neurons_as_vector):
        for k in range(len(self.layers)):
            weights = self.layers[k].next_weights
            if(type(weights) != str and weights != 'end'):
                for i in range(len(weights)):
                    for j in range(len(weights[i])):
                        self.layers[k].next_weights[i][j] = neurons_as_vector.pop(0)
                    

    #inputs-> the attributes of the training data
    #outputs-> the actual classes for each attribute row
    def fitness(self, inputs, outputs):
        diff = 0
        for i in range(len(inputs)):
            guess = self.predict(inputs[i])
            #guess -> MLP output of size of output layer
            for g in range(len(guess)):
                diff += (outputs[i][g] - guess[g])**2
        diff /= len(inputs)
        self.individual_fitness = 1/diff
        return self.individual_fitness
