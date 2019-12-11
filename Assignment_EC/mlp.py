'''
This file contains the MLP structure which are members of our algorithms populations
This is a feed forward neural network with no learning methods as that will be handled through evolution in this implementation
'''

from layer import Layer
import numpy as np
import math
import random

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

    #takes an input point and feeds it forward to return a vector of outputs based on its layers
    def predict(self, point, regression='default', backprop = False):
        dat = np.transpose([point])
        self.layers[0].set_nodes(dat,persistent=backprop)
        for i in range(len(self.layers)-1):
            nxt_lyr = self.layers[i].feed_forward_sigmoid()
            self.layers[i+1].set_nodes(nxt_lyr,persistent=backprop)
        if(regression == 'default'):
            if(self.layers[-1].get_layer_size() == 1):
                regression = True
            else:
                regression = False

        if(regression):
            nxt_lyr = self.layers[-2].feed_forward()
        else:
            nxt_lyr = self.layers[-2].feed_forward_sigmoid()
        self.layers[-1].set_nodes(nxt_lyr,persistent=backprop)

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
        # print(neurons_as_vector[0])
        old_layers = self.layers
        for k in range(len(self.layers)):
            weights = self.layers[k].next_weights
            if(type(weights) != str and weights != 'end'):
                for i in range(len(weights)):
                    for j in range(len(weights[i])):
                        self.layers[k].next_weights[i][j] = neurons_as_vector.pop(0)
        # print("updated layers ", old_layers[0].next_weights != self.layers[0].next_weights)


    #inputs-> the attributes of the training data
    #outputs-> the actual classes for each attribute row
    def fitness(self, inputs, outputs):
        diff = 0
        for i in range(len(inputs)):
            guess = self.predict(inputs[i])
            #guess -> MLP output of size of output layer
            if len(guess) > 1:
                for g in range(len(guess)):
                    # diff += (outputs[i][g] - guess[g])**2
                    #diff += abs(outputs[i][g] - guess[g])

                    #----incentive------
                    if(outputs[i][g] == 1):
                        diff += (outputs[i][g] - guess[g])**2
                    else:
                        diff += ((outputs[i][g] - guess[g])**2)/50
                    #----------
            else:
                diff += (outputs[i] - guess[0])**2

        diff /= len(inputs)
        self.individual_fitness = 1/(diff + 0.001)
        return self.individual_fitness

    def fitness_with_f1(self, inputs, outputs):
        diff = 0
        for i in range(len(inputs)):
            guess = self.predict(inputs[i])
            #guess -> MLP output of size of output layer
            if len(guess) > 1:
                maximum = 0
                idxmax = 0
                for g in range(len(guess)):

                    if(guess[g] >= maximum):
                        maximum = guess[g]
                        idxmax = g


                if(outputs[i][idxmax]==1):
                    pass
                else:
                    diff+=1
            else:
                diff += (outputs[i] - guess[0])**2

        diff /= len(inputs)
        self.individual_fitness = 1/(diff + 0.001)
        return self.individual_fitness

    def binary_cross_entropy(self, inputs, outputs):
        error = 0
        error_arr = [0]*len(inputs[0])
        for i in range(len(inputs)):
            guess = self.predict(inputs[i])
            for n in range(len(guess)):
                yi = guess[n]
                y = outputs[i][n]

                error += y*ln(yi) + (1-y)*ln(1-yi)
                error_arr[n] += y*ln(yi) + (1-y)*ln(1-yi)

        error/=(len(inputs)*len(inputs[0]))
        error_arr/=len(inputs)
        return 0-error, 0-error_arr

    def mse(self, inputs, outputs):
        diff = 0
        diff_arr = [0]*len(inputs[0])
        for i in range(len(inputs)):
            guess = self.predict(inputs[i])
            for j in range(len(guess[i])):
                diff += (outputs[i][j] - guess[j])**2
                diff_arr[j]+=(outputs[i][j] - guess[j])**2

        diff /= (len(inputs)*len(inputs[0]))
        diff_arr /= len(inputs)
        return diff, diff_arr

    def backprop(self, test, target, learning_rate, loss="mse"):

        for iters in range(10000):

            for layer in self.layers:
                layer.nodes_persistent*=0

            batch = []
            batch_targets = []
            for rand in range(int(len(test)/4)):
                rand = random.randrange(0,len(test))
                batch.append(test[rand])
                batch_targets.append(target[rand])

            #guess = self.predict(test[0]) * 0
            cum_err = len(batch_targets[0])*[0]
            guesses = []
            for train in range(len(batch)):
                #guess += self.predict(batch[train], backprop = True)
                guess = self.predict(batch[train], backprop = True)
                guesses.append(guess)
                tot, mse_arr = mse_dir([guess], [batch_targets[train]])
                for node in range(len(cum_err)):
                    cum_err[node] += mse_arr[node]/len(batch)
            #print(guesses)
            #print(batch_targets)
            #print(cum_err)

            layer_num = len(self.layers)

            for layer in range(layer_num-1): #we wont check first layer as there isn't anything that feeds into it
                curr_layer = layer_num-layer-1 #start at output and work backwards

                #print(self.layers[curr_layer].nodes_persistent)
                #guess = self.layers[curr_layer].nodes_persistent/len(batch)
                #guess = guess.transpose()[0].tolist()
                #print(guess)

                prev_layer = self.layers[curr_layer-1]
                #print(prev_layer.next_weights)

                next_error = len(prev_layer.nodes) * [0]
                for node in range(len(prev_layer.next_weights)):

                    if(layer == 0): #output layer, sigmoid was not calculated
                        sig = 1
                    else:
                        #print(self.layers[curr_layer].nodes_persistent)
                        node_val = self.layers[curr_layer].nodes_persistent[node]/len(batch)
                        sig = node_val*(1-node_val)

                    for weight in range(len(prev_layer.next_weights[node])):

                        #adj = cum_err[node] * sig * prev_layer.next_weights[node][weight]
                        adj = cum_err[node] * sig * prev_layer.nodes_persistent[weight]/len(batch)

                        #next_error[weight] += prev_layer.nodes_persistent[weight]/len(batch) * cum_err[node] * sig
                        next_error[weight] += prev_layer.next_weights[node][weight] * cum_err[node] * sig

                        prev_layer.next_weights[node][weight] -= learning_rate * adj

                cum_err = next_error.copy()
                #print(prev_layer.next_weights)



def ln(x):
    return math.log(x)

def mse(inputs, outputs):
    diff = 0
    diff_arr = [0]*len(inputs[0])
    for i in range(len(inputs)):
        for j in range(len(inputs[i])):
            diff += (inputs[i][j] - outputs[i][j])**2
            diff_arr[j]+=(inputs[i][j] - outputs[i][j])**2

    diff /= (len(inputs)*len(inputs[0]))
    for i in range(len(diff_arr)):
        diff_arr[i] /= len(inputs)
    return diff, diff_arr

def mse_dir(inputs, outputs):
    diff = 0
    diff_arr = [0]*len(inputs[0])
    for i in range(len(inputs)):
        for j in range(len(inputs[i])):
            diff += (inputs[i][j] - outputs[i][j])
            diff_arr[j]+=(inputs[i][j] - outputs[i][j])

    diff /= (len(inputs)*len(inputs[0]))
    for i in range(len(diff_arr)):
        diff_arr[i] /= len(inputs)
    return diff, diff_arr


if __name__ == "__main__":
    # training = [[0,0,0],[0,1,1],[1,0,1],[1,1,0]]
    # training2 = [[0,0,0],[0,1,1],[1,0,1],[1,1,0],[1,0,0],[0,0,1]]
    # outputs2 = [[1,1,1],[1,0,0],[0,1,0],[0,0,1],[0,1,1],[1,1,0]]
    #
    # mlp = MLP([3,4,3])
    # guess = mlp.predict(training2[0], backprop = True)
    # print(mlp.predict(training2[0]))
    # mlp.backprop(training2,outputs2,0.5)
    #
    #
    # print(mlp.predict(training2[0]))
    # print(mlp.predict(training2[1]))
    # print(mlp.predict(training2[2]))

    training2 = [[0,0],[0,1],[1,0],[1,1]]
    training2 = [[0,0],[0,1],[1,0],[1,1]]
    outputs2 = [[0],[1],[1],[0]]

    mlp = MLP([2,4,1])
    guess = mlp.predict(training2[0], backprop = True)
    print(mlp.predict(training2[0]))
    mlp.backprop(training2,outputs2,0.5)


    print(mlp.predict(training2[0]))
    print(mlp.predict(training2[1]))
    print(mlp.predict(training2[2]))
    print(mlp.predict(training2[3]))
