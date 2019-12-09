import numpy as np
import random


class Layer():

    def __init__(self, number_of_nodes, next_layer_number='none', previous_layer_number='none'):

        self.nodes = np.zeros((number_of_nodes,1))

        if(type(next_layer_number) != str and next_layer_number!='none'):
            self.next_weights = np.zeros((next_layer_number,number_of_nodes))

            for row in range(len(self.next_weights)):
                for col in range(len(self.next_weights[row])):
                    self.next_weights[row][col] = random.random()
        else:
            self.next_weights = next_layer_number

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

    def get_layer_size(self):
        return self.nodes.shape[0]

    def set_nodes(self, vector):
        if(type(vector) == str):
            for row in range(len(self.nodes)):
                for col in range(len(self.nodes[row])):
                    self.nodes[row][col] = random.random()
        elif(self.nodes.shape == vector.shape):
            self.nodes = vector
        else:
            raise Exception('Shape Mismatch: given vector does not fit input layer. Size given:' + str(vector.shape[0]) + '. Required:'+ str(self.nodes.shape[0]))

    def adjust_nodes(self, adjustment):
        if(type(adjustment) == str and adjustment == 'random'):
            for row in range(len(self.nodes)):
                for col in range(len(self.nodes[row])):
                    self.nodes[row][col] = random.random()
        else:
            self.nodes += adjustment

    def feed_forward(self):
        if(type(self.next_weights)!=str):
            return np.dot(self.next_weights , self.nodes)
        else:
            return 'end'

    def feed_forward_sigmoid(self):
        if(type(self.next_weights)!=str):
            temp = np.dot(self.next_weights , self.nodes)
            exp = np.exp(-temp)
            exp_1 = np.add(exp, 1)
            temp = 1/exp_1
            return temp
        else:
            return 'end'
