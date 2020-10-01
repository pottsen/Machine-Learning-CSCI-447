import numpy as np

"""
Implement a radial basis function neural network with an arbitrary number of inputs, an arbitrary number of Gaussian basis functions, and an arbitrary number of outputs. As with the feedforward network, your program should accept the number of inputs, Gaussians, and outputs. It is your choice which output activation function is used, but it should be appropriate to the data set being trained.
"""

class RBN():
    #input-> #data_as_2dList, possible_outputs, number_of_hidden_nodes
    def __init__(self, data, output, gaussian_function_type, centers):
        
        self.data = np.array(data)
        self.learning_rate = 0.01
        self.number_of_nodes = len(centers)
        self.centers = np.array(centers)
        self.rbf_layer = np.random.rand(1,len(centers))
        self.outputs = np.zeros(len(output))
        self.outputs.shape = (1,len(output))
        self.weight_matrix = np.random.rand(len(output), len(centers))
        self.weight_matrix /= len(centers)**2
        # print(self.weight_matrix)
        self.d_max = self.distance_max(self.centers)
        self.stdev = float(self.d_max) / (2 * len(self.centers))**0.5
        self.errors = np.zeros(self.outputs.shape)
        self.cumulative_errors = np.zeros(self.outputs.shape)
        self.cumulative_targets = np.zeros(self.outputs.shape)
        self.cumulative_outputs = np.zeros(self.outputs.shape)
        self.cumulative_inputs = np.zeros((1,len(self.data[0])-1))
        self.cumulative_weight_matrix = np.zeros(self.weight_matrix.shape)
        self.cumulative_rbf_layer = np.zeros(self.rbf_layer.shape)

    def __str__(self):
        stringify = "Weights:\n"
        stringify += str(self.weight_matrix)
        stringify += "\nOutputs\n"
        stringify += str(self.outputs)
        return stringify
    
    def train(self):
        temp = []
        equal = False
        iterations = 0
        while(not equal) and iterations < 1000:
            tempWM = np.copy(self.weight_matrix)
            # tempLM = self.linear_matrix
            self.network_train_iteration()
            #print("outputs\n",self.outputs)
            print(self)
            iterations +=1
            print("Iteration ", iterations)
            equal = True

            # print("temp WM \n", tempWM)
            # print("WM\n", self.weight_matrix)
            if(not np.array_equal(self.weight_matrix, tempWM)):
                    equal = False
            # if(not np.array_equal(self.linear_matrix,tempLM):
            #         equal = False
            print(equal)

    def network_train_iteration(self):
        self.cumulative_errors = np.multiply(self.cumulative_errors,0)
        # print("cum error\n", self.cumulative_errors.shape)
        self.cumulative_targets = np.multiply(self.cumulative_errors,0)
        # print("cum target\n", self.cumulative_targets.shape)
        self.cumulative_outputs = np.multiply(self.cumulative_outputs,0)
        # print("cum target\n", self.cumulative_targets.shape)
        self.cumulative_inputs = np.multiply(self.cumulative_inputs,0)
        self.cumulative_weight_matrix = np.multiply(self.cumulative_weight_matrix,0)
        self.cumulative_rbf_layer = np.multiply(self.cumulative_rbf_layer,0)

            
        #for every data point (vector)
        for d in self.data:

            self.target = d[0] #first index is the class
            inputs = np.stack(d[1:])
            centers_no_class = self.centers[:,1:]
            #center_classes = self.centers[:,0]

              
            # self.cumulative_inputs += inputs

            # self.rbf_layer = np.dot(inputs, self.weight_matrix)
            self.feed_forward(inputs, centers_no_class)

            #calculate errors Total and by class
            # self.cumulative_update(actual, self.outputs)
            if len(self.outputs) == 1:  #regression problem
                # print("here")
                self.errors = (self.target - self.outputs)
                # print("errors shape\n", self.errors, self.errors.shape)
                # print("outputs \n", self.outputs, self.outputs.shape)
                # print("target vector\n", self.target)

            else:  #classification
                # print("AQUI")
                self.target_vector = np.zeros((self.outputs.shape))
                # print("target vector\n", self.target_vector.shape)
                self.target_vector[(int(self.target)-1)] = 1    #ex: actual = 3 -->   [0, 0, 1]
                # print(self.target_vector)
                self.errors = np.subtract(self.target_vector, self.outputs) #np array
                # print("errors shape\n", self.errors, self.errors.shape)
                # print("outputs \n", self.outputs, self.outputs.shape)

            self.backprop_single_peter()
            # print(self)
        # take average of all cumulative data for use in backprop calculations
        # self.cumulative_errors /= len(self.data)
        # self.cumulative_outputs /= len(self.data)
        # self.cumulative_targets /= len(self.data)
        # self.cumulative_inputs /= len(self.data)
        # self.cumulative_weight_matrix /= len(self.data)
        # self.cumulative_rbf_layer /= len(self.data)

        # self.backprop_peter()
        # self.backpropagate(layer_target_num - 1, self.errors)

    def cumulative_update(self,actual, outputs): #(MSE)
        self.errors = np.zeros(self.outputs.shape)
        # print(len(self.outputs[0]))
        if len(self.outputs[0]) == 1:  #regression problem
            # sum errors
            # print("HERE")
            self.errors += (actual - outputs)
            self.cumulative_errors -= (outputs - actual)
            #sum the output vector values for use in backprop
            self.cumulative_outputs += outputs
            #sum the target vector values for use in backprop
            self.cumulative_targets += actual
            #sum the weight matrix values for use in backprop
            self.cumulative_weight_matrix += self.weight_matrix
            #sum rbf layer
            self.cumulative_rbf_layer += self.rbf_layer

        else:  #classification
            # print("AQUI")
            # if(type(actual) == int or type(actual) == float ):
            actual_vector = np.zeros((len(outputs[0],1)))
            print("actual vector\n", actual_vector.shape)
            actual_vector[int(actual) - 1][0] = 1    #ex: actual = 3 -->   [0, 0, 1]
            print(actual_vector)

            # else:
            #     actual_vector = actual

            # print("Actual targets\n", actual_vector)
            # print("outputs\n", outputs)
            #sum errors
            self.errors = np.subtract(actual_vector, outputs) #np array
            self.cumulative_errors -= np.subtract(outputs, actual_vector) #np array
            #sum the output vector values for use in backprop
            self.cumulative_outputs += outputs
            #sum the target vector values for use in backprop
            self.cumulative_targets += actual_vector
            #sum the weight matrix values for use in backprop
            self.cumulative_weight_matrix += self.weight_matrix
            #sum rbf layer
            self.cumulative_rbf_layer += self.rbf_layer

    def feed_forward(self, inputs, centers_no_class):
        # print("centers_no_class\n", centers_no_class, centers_no_class.shape)
        deltas = np.subtract(inputs, centers_no_class)
        # print("deltas\n", deltas, deltas.shape)
        deltas_squared = np.square(deltas)
        # print("deltas squared \n", deltas_squared, deltas_squared.shape)
        exp_inside = np.divide(deltas_squared, (-2 * self.stdev**2))
        # print("exp_inside\n", exp_inside.shape)
        sum_exp = np.zeros((len(exp_inside),1))
        # print("sum exp\n", sum_exp.shape)
        for i in range(len(exp_inside)):
            sum_exp[i,0] = np.sum(exp_inside[i])
        # print("sum exp\n", sum_exp)
        exp = np.exp(sum_exp)
        # print("exp\n", exp.shape)
        self.rbf_layer = exp
        # print("rbf layer \n", self.rbf_layer.shape)
        

        #now multiply RBF layer by weights to get output
        self.outputs = np.dot(self.weight_matrix, self.rbf_layer)
        self.outputs = np.divide(self.outputs, self.number_of_nodes)
        # print("outputs", self.outputs.shape)
        

    def distance_max(self, centers):
        dist_max = 0
        for i in range(len(centers)-1):
            j = i + 1
            while j < len(centers):
                distance = np.sum((centers[i]-centers[j])**2)
                if distance > dist_max:
                    dist_max = distance
                j+=1
        return dist_max

    def backprop_peter(self):
        #update WMs. All cumulative values were averaged before calling back prop
        # print("Cumulative RBF\n", self.cumulative_rbf_layer.shape)
        # print("Errors \n", self.errors.shape)
        # print("Errors\n", self.errors)
        delta_WM = self.learning_rate * np.dot(np.transpose(self.cumulative_rbf_layer), self.errors)
        # print("Delta WM\n", delta_WM)
        self.weight_matrix += delta_WM

    def backprop_single_peter(self):
        #update WMs. All cumulative values were averaged before calling back prop
        # print("rbf shape\n", self.rbf_layer.shape)
        # print("errors shape\n", self.errors.shape)
        # print("WM shape\n", self.weight_matrix.shape)
        delta_WM = self.learning_rate * np.dot(self.errors, np.transpose(self.rbf_layer))
        # print("Delta WM\n", delta_WM)
        self.weight_matrix += delta_WM


    #takes a list of training points and returns a list of tuples corresponding to them (actual class/value, guessed class/value)
    def classify(self, test_data):
        centers_no_class = self.centers[:,1:]
        center_classes = self.centers[:,0]


        guesses = []
        for i in test_data:
            act = i[0]
            data = i[1:]
            self.feed_forward(data, centers_no_class)
            out = self.outputs

            if(len(out)==1):#regression
                out = out[0]
            else:#classification
                out = np.argmax(self.outputs[0])+1 #index of max (self.outputs)
            
            guesses.append([act,out])
            
        return guesses
        