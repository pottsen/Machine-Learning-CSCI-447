'''
This is a helper class of MLP to handle each layer seperately with its weight matrix

'''

import numpy as np
import random


class Layer():

    def __init__(self, number_of_nodes, next_layer_number='none', previous_layer_number='none'):

        self.nodes = np.zeros((number_of_nodes,1))
        self.nodes_persistent = np.zeros((number_of_nodes,1))

        #initialize feed forward weights randomly
        if(type(next_layer_number) != str and next_layer_number!='none'):
            self.next_weights = np.zeros((next_layer_number,number_of_nodes))

            for row in range(len(self.next_weights)):
                for col in range(len(self.next_weights[row])):
                    self.next_weights[row][col] = random.random()
        else:
            self.next_weights = next_layer_number

        #initialize feed back weights randomly (not used for assignment 4)
        if(type(previous_layer_number) != str and previous_layer_number!='none'):
            self.prev_weights = np.zeros((previous_layer_number,number_of_nodes))

            for row in range(len(self.prev_weights)):
                for col in range(len(self.prev_weights[row])):
                    self.prev_weights[row][col] = random.random()
        else:
            self.prev_weights = previous_layer_number

    def __str__(self):
        string = ""

        for i in self.nodes:
            string += "[{0:.3f}]".format(i[0]) + '\n'

        return string

    #returns number of nodes a layer has
    def get_layer_size(self):
        return self.nodes.shape[0]

    #sets the layers nodes to be values in vector
    def set_nodes(self, vector, persistent = False):
        if(type(vector) == str):
            for row in range(len(self.nodes)):
                for col in range(len(self.nodes[row])):
                    self.nodes[row][col] = random.random()
        elif(self.nodes.shape == vector.shape):
            self.nodes = vector
            if(persistent):
                self.nodes_persistent+=vector
        else:
            raise Exception('Shape Mismatch: given vector does not fit input layer. Size given:' + str(vector.shape[0]) + '. Required:'+ str(self.nodes.shape[0]))

    #change nodes by a value (adjustment) different than set nodes because initial node values matter
    def adjust_nodes(self, adjustment):
        if(type(adjustment) == str and adjustment == 'random'):
            for row in range(len(self.nodes)):
                for col in range(len(self.nodes[row])):
                    self.nodes[row][col] = random.random()
        else:
            self.nodes += adjustment

    #takes the values of its nodes and dot products it with its weights to send to the next layer
    def feed_forward(self):
        if(type(self.next_weights)!=str):
            return np.dot(self.next_weights , self.nodes)
        else:
            return 'end'

    #takes the values of its nodes and dot products it with its weights to send to the next layer (output is normalized to be between 0-1)
    def feed_forward_sigmoid(self):
        if(type(self.next_weights)!=str):
            temp = np.dot(self.next_weights , self.nodes)
            exp = np.exp(-temp)
            exp_1 = np.add(exp, 1)
            temp = 1/exp_1
            return temp
        else:
            return 'end'
