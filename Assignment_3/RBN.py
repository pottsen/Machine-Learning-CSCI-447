import numpy as np

"""
Implement a radial basis function neural network with an arbitrary number of inputs, an arbitrary number of Gaussian basis functions, and an arbitrary number of outputs. As with the feedforward network, your program should accept the number of inputs, Gaussians, and outputs. It is your choice which output activation function is used, but it should be appropriate to the data set being trained.
"""

class RBN():
    #input-> #data_as_2dList, possible_outputs, number_of_hidden_nodes
    def __init__(self, data, output, number_of_nodes, gaussian_function_type, centers):
        
        self.data = data
        self.learning_rate = 0.1
        self.rbf_layer = np.random.rand(1,number_of_nodes)
        self.outputs = np.zeros(len(output))
        self.outputs.shape = (1,len(output))
        self.weight_matrix = np.random.rand(len(self.data[0])-1, number_of_nodes)
        print("weight matrix shape ", self.weight_matrix.shape)
        self.linear_matrix = np.random.rand(number_of_nodes, len(output)
        print("linear matrix shape ", self.linear_matrix.shape)
        self.centers = centers
        self.d_max = self.distance_max(self.centers)
        self.stdev = self.d_max / (2 * len(self.centers)**0.5)

    def train(self):
        temp = []
        equal = False
        iterations = 0
        while(not equal) and iterations < 10000:
            tempWM = self.weight_matrix
            # tempLM = self.linear_matrix
            self.network_train_iteration()
            #print("outputs\n",self.outputs)
            print(self)
            iterations +=1
            print("Iteration ", iterations)
            equal = True

            if(not np.array_equal(self.weight_matrix,tempWM):
                    equal = False
            # if(not np.array_equal(self.linear_matrix,tempLM):
            #         equal = False

    def network_train_iteration(self):
        self.errors = np.zeros(self.outputs.shape)
        self.cumulative_targets = np.zeros(self.outputs.shape)
        self.cumulative_outputs = np.zeros(self.outputs.shape)
        self.cumulative_inputs = np.zeros((1,len(self.data[0])-1))
        self.cumulative_weight_matrix = np.zeros(self.weight_matrix.shape)
        self.cumulative_rbf_layer = np.zeros(self.rbf_layer.shape)

            
        #for every data point (vector)
        for d in self.data:

            actual = d[0] #first index is the class
            # centers_no_class = self.centers([:][1:])
            centers_no_class = self.centers[:,1:]
            inputs = np.transpose(d[1:])  
            self.cumulative_inputs += inputs

            # self.rbf_layer = np.dot(inputs, self.weight_matrix)
            self.rbf_layer = centers_no_class - inputs
            self.rbf_layer = np.power(self.rbf_layer, 2)
            self.rbf_layer = np.sum(self.rbf_layer, 1)
            self.rbf_layer = self.rbf_layer / (-2 * self.stdev**2)
            self.rbf_layer = np.exp(self.rbf_layer)

            #now multiply RBF layer by weights to get output
            self.outputs = np.dot(self.weight_matrix, self.rbf_layer)
            
            #START BACK HERE
            #calculate errors Total and by class
            self.cumulative_update(actual, self.outputs)
        
        # take average of all cumulative data for use in backprop calculations
        self.errors /= len(self.data)
        self.cumulative_outputs /= len(self.data)
        self.cumulative_targets /= len(self.data)
        self.cumulative_inputs /= len(self.data)
        self.cumulative_weight_matrix /= len(self.data)
        self.cumulative_rbf_layer /= len(self.data)

        self.backprop_peter()
        # self.backpropagate(layer_target_num - 1, self.errors)

    def cumulative_update(self,actual, outputs): #(MSE)
        
        if len(outputs) == 1:  #regression problem
            # sum errors
            self.errors += (outputs - actual)
            #sum the output vector values for use in backprop
            self.cumulative_outputs += outputs
            #sum the target vector values for use in backprop
            self.cumulative_targets += actual
            #sum the weight matrix values for use in backprop
            self.cumulative_weight_matrix += self.weight_matrix
            #sum rbf layer
            self.cumulative_rbf_layer += self.cumulative_rbf_layer

        else:  #classification
            if(type(actual) == int or type(actual) == float ):
                actual_vector = np.zeros((1,len(outputs[0])))
                actual_vector[0][int(actual) - 1] = 1    #ex: actual = 3 -->   [0, 0, 1]

            else:
                actual_vector = actual

            #sum errors
            self.errors += np.subtract(outputs, actual_vector) #np array
            #sum the output vector values for use in backprop
            self.cumulative_outputs += outputs
            #sum the target vector values for use in backprop
            self.cumulative_targets += actual_vector
            #sum the weight matrix values for use in backprop
            self.cumulative_weight_matrix += self.weight_matrix
            #sum rbf layer
            self.cumulative_rbf_layer += self.cumulative_rbf_layer


    def distance_max(centers):
        d_max = 0
        for i in range(len(centers)-1):
            j = i + 1
            while j < range(len(centers):
                distance = np.sum((centers[i]-centers[j])**2)
                if distance > d_max:
                    d_max = distance
                j+=1

    def backprop_peter(self):
        #update WMs. All cumulative values were averaged before calling back prop
        delta_WM = self.learning_rate * self.errors * self.cumulative_rbf_layer

        self.weight_matrix -= delta_WM

    #momentum can be used here
    def delta_weight_matrix(self, weight_matrix, regularizer, error_vector, avg_layer_values, previous_deltaWM):
        #uses equation delta_WM_i = learning_rate * Layer_Error_Vector * (avg_layer_output*(1-avg_layer_output)) * Avg_output_previous_layer_vector
        if (not self.momentum):
            error_regularizer = error_vector * regularizer
            delta_weight_matrix = self.learing_rate * np.dot(np.transpose(avg_layer_values), error_regularizer)
    """
    def gaussian_rbf(self, center, point, stdev):
        gaussian_sum = 0
        # not sure if for loop is needed
        return np.sum(np.exp(-(point-center)**2/(2*stdev**2)))
        # for i in range(len(point)):
        #     gaussian_sum += np.exp(-(point[i]-center[i])**2/(2*stdev**2))

        # return gaussian_sum
    """
