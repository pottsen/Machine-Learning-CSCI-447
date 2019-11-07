import numpy as np



class MLP():
    #input-> #data_as_2dList, possible_outputs, number_of_hidden_layers, number_of_hidden_nodes_in_each_layer
    def __init__(self, data, output, number_of_layers, number_of_nodes, momentum):
        # flag 
        self.momentum = bool(momentum)
        self.momentum_factor = 0.3
        self.learing_rate = 0.1
        self.data = data
        self.outputs = np.zeros(len(output))
        self.outputs.shape = (1,len(output))
        #self.class_names = output #should this be a list of the possible classes????????

        if (len(number_of_nodes) != number_of_layers):
                raise Exception ("we need to know how many nodes are in each hidden layer")

        #list of np arrays - every np array is a hidden layer
        self.hidden_layers = []
        for layer in range(number_of_layers):
            self.hidden_layers.append(np.random.rand(1,number_of_nodes[layer]))

        #make an list of np wieght matricies
        self.weight_matricies = []
        #make a list of np matrices for storing previous WM changes for momentum
        self.previous_WM_delta = []
        #add matrices to the list
        if number_of_layers > 0:
            #input and first hidden layer
            self.weight_matricies.append(np.random.rand(len(self.data[0])-1, number_of_nodes[0]))
            self.previous_WM_delta.append(np.zeros((len(self.data[0])-1, number_of_nodes[0])))
            #current to next hidden layer
            for i in range(number_of_layers-1):
                layer = np.random.rand(number_of_nodes[i], number_of_nodes[i+1])
                layer_zeros = np.zeros((number_of_nodes[i], number_of_nodes[i+1]))
                self.weight_matricies.append(layer)
                self.previous_WM_delta.append(layer_zeros)
            self.weight_matricies.append(np.random.rand(number_of_nodes[-1], len(output)))
            self.previous_WM_delta.append(np.zeros((number_of_nodes[-1], len(output))))
        else:
            #if no hidden layers inputs and outputs
            self.weight_matricies.append(np.random.rand(len(self.data[0])-1, len(output)))
            self.previous_WM_delta.append(np.zeros((len(self.data[0])-1, len(output))))

        # print("# weight matrices ", len(self.weight_matricies))
        # print(self.weight_matricies)
        # print("prev weight matrices ", len(self.previous_WM_delta))
        
    def __str__(self):
        stringify = "NETWORK:"
        for i in range(len(self.weight_matricies)):
            stringify+= "\nWEIGHTS:\n"+str(self.weight_matricies[i])
            if i == len(self.weight_matricies) - 1:
                pass #stringify += "\nOUTPUTS:\n"+str(self.outputs)
            else:
                stringify += "\nHIDDEN LAYER\n"+str(self.hidden_layers[i])
        stringify += "\nEND NETWORK"
        return stringify

    def train(self):
        temp = []
        equal = False
        iterations = 0
        while(not equal) and iterations < 10000:
            temp = []
            for i in self.weight_matricies:
                temp.append(np.copy(i))
            self.network_train_iteration()
            # print("outputs\n",self.outputs)
            # print(self)
            iterations +=1
            print("Iteration ", iterations)
            equal = True
            for i in range(len(self.weight_matricies)):
                self.weight_matricies[i] =self.weight_matricies[i].round(decimals = 4)
                if(not np.array_equal(self.weight_matricies[i],temp[i])):
                    equal = False


    def sigmoidify_layer(self, layer):
        for i in range(len(layer)):
            layer[i] = self.sigmoid(layer[i])
        return layer

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))


    def network_train_iteration(self):
        layer_target_num = 0

        self.errors = np.zeros(self.outputs.shape)
        self.cumulative_targets = np.zeros(self.outputs.shape)
        self.cumulative_outputs = np.zeros(self.outputs.shape)
        self.cumulative_inputs = np.zeros((1,len(self.data[0])-1))
        
        self.cumulative_hidden_layers = []
        for layer in range(len(self.hidden_layers)):
            self.cumulative_hidden_layers.append(np.zeros((1,len(self.hidden_layers[layer][0]))))
            
        #for every data point (vector)
        for d in self.data:

            actual = d[0] #first index is the class

            curr_layer = np.transpose(d[1:])  #for any two adjacent layers, curr_layer is the input layer
            self.cumulative_inputs += curr_layer

            layer_target_num = 1 # the layer (not exclusively hidden) the local output is targeting

            #for every weight matrix (# of hidden layers + 1)
            for i in range(len(self.hidden_layers)+1):
                #output layer:
                if(len(self.hidden_layers) < layer_target_num): # the current layers output will be the classification/output layer
                    next_layer = self.outputs   #target layer of curr_layer
                    weights = self.weight_matricies[i]  #weights mapping from curr_layer to target_layer
                    curr_layer = self.feed_forward_layer(curr_layer,next_layer,weights)
                    if (len(self.outputs[0]) >1): #this is a classifcation problem, and we want our outputs to be between 0-1
                        curr_layer = self.sigmoidify_layer(curr_layer)
                    self.outputs = curr_layer

                #hidden layer:
                else: #the output of the current layer is the input to another hidden layer
                    next_layer = self.hidden_layers[layer_target_num-1]   #target layer of curr_layer
                    weights = self.weight_matricies[i]  #weights mapping from curr_layer to target_layer
                    curr_layer = self.feed_forward_layer(curr_layer,next_layer,weights)
                    curr_layer = self.sigmoidify_layer(curr_layer)
                    self.hidden_layers[layer_target_num-1] = curr_layer

                layer_target_num += 1 #iterate the target layer to next layer
            #print("hidden layer")
            #print(self.hidden_layers)
            # print("outputs")
            # print(self.outputs)

            #calculate errors Total and by class
            self.errors += self.cumulative_update(actual, self.outputs)
        
        # take average of all cumulative data for use in backprop calculations
        self.errors /= len(self.data)
        self.cumulative_outputs /= len(self.data)
        self.cumulative_targets /= len(self.data)
        self.cumulative_inputs /= len(self.data)
        for i in range(len(self.cumulative_hidden_layers)):
            self.cumulative_hidden_layers[i] /= len(self.data)

        self.backprop_peter()
        # self.backpropagate(layer_target_num - 1, self.errors)


    #calculating node values for a hidden layer
    def feed_forward_layer(self,layer1,layer2,layer1_weights):
        layer2 = np.dot(layer1,layer1_weights)
        layer2.shape = (1,len(layer1_weights[0]))
        return layer2

    #inputs-> actual class index,  output array values


    #input example-> 3, [.8,.2,.6]
    #actual is either the class index from 1-n or the value of the discrete class
    def cumulative_update(self,actual, outputs): #(MSE)
        
        if len(outputs[0]) == 1:  #regression problem
            errors = [ (outputs[0] - actual)]
            #sum the output vector values for use in backprop
            self.cumulative_outputs += outputs

            #sum the target vector values for use in backprop
            self.cumulative_targets += actual

            #sum the hidden layer values for use in backprop
            for i in range(len(self.hidden_layers)):
                self.cumulative_hidden_layers[i] += self.hidden_layers[i]

        else:  #classification
            if(type(actual) == int or type(actual) == float ):
                actual_vector = np.zeros((1,len(outputs[0])))
                actual_vector[0][int(actual) - 1] = 1    #ex: actual = 3 -->   [0, 0, 1]

            else:
                actual_vector = actual


            #errors = (guessed - actual)^2
            errors = np.subtract(outputs, actual_vector) #np array

            #sum the output vector values for use in backprop
            self.cumulative_outputs += outputs

            #sum the target vector values for use in backprop
            self.cumulative_targets += actual_vector

            #sum the hidden layer values for use in backprop
            for i in range(len(self.hidden_layers)):
                self.cumulative_hidden_layers[i] += self.hidden_layers[i]

        return np.transpose(np.transpose(errors))

        # for i in range(len(outputs)):
        #     if (actual-1) == i:
        #         # DO WE NEED THIS? SHOULD JUST SUM ALL E_i
        #         # self.total_error += 0.5 * (1-outputs[i])**2
        #         self.errors[i] += 0.5 * (1-outputs[i])**2
        #         self.error_partials[i] += -(1-outputs[i])
        #     else:
        #         # DO WE NEED THIS? SHOULD JUST SUM ALL E_i
        #         # self.total_error += 0.5 * (0-outputs[i])**2
        #         self.errors[i] += 0.5 * (0-outputs[i])**2
        #         self.error_partials[i] += -(0-outputs[i])

    """
    def backpropagate(self, layer, errors):
        #for every weight matrix (# of hidden layers + 1)

        feed_back_values = np.transpose(self.cumulative_targets)
        for i in range(len(self.hidden_layers)+1):

            if(layer > 1): #we are at least past the first hidden layer, so we need to backpropogate the "actual" values of the previous hidden layer
                #reverse_weights = 1/self.weight_matricies[layer-1]
                print(feed_back_values.shape)
                reverse_weights = self.weight_matricies[layer-1]
                feed_back_values = np.dot(reverse_weights, feed_back_values) #feed_back_values are actual values of the hidden layer

            #self.backpropagate_layer(layer, errors)
            self.backprop_peter()

            # if (layer> 1):
            #     errors = prev_error


            layer -= 1 #iterate the target layer to next layer

    
    def backpropagate_layer(self,layer_no,errors):

        # ∇E(h,n) = 1/n * Σ (g_n - a_n) * regularizer' * activation_n_h
        # errors = [1/n * Σ (g_n0 - a_n0), 1/n * Σ (g_n1 - a_n1), ...]
        # regulizer is the derivative of the node normalization function
        # n is the output node
        # h is a hidden layer node
        # activation_n_h is the activation from node h to node n

        # partial_totalE_Wij = pE_t/pO_i * pO_i/pNet_i * pNet_i/pW_ijhttp://csci491-01.cs.montana.edu/~w32g348/www/montanahang/

        weight_gradient = np.zeros((self.weight_matricies[layer_no-1].shape))

        #output layer:
        # if(len(self.hidden_layers) < layer_no): # the current layers output will be the classification/output layer
        #     if (len(self.cumulative_outputs[0]) >1): #this is a classifcation problem, and we want to multiply by the derivative of our sigmoid
        #         regularizer = self.cumulative_outputs * (1-self.cumulative_outputs)
        #     else:
        #         regularizer = 1

        #     # for i in range(len(self.weight_matricies[layer_no-1][0])):
        #     #     matrix = self.weight_matricies[layer_no-1]
        #     #     i = np.transpose(np.transpose(matrix)[i])
        #     #     i.shape = (len(i),1)                
        #     #     weight_gradient += i * (errors * regularizer)
            
        #     i = self.cumulative_hidden_layers[layer_no-1-1] # -1 for previous layer and -1 becauses hidden layers at idx 0 is actually the network at ids 1
        #     weight_gradient += i * (errors * regularizer)   
        if(layer_no == 0):
            regularizer = self.cumulative_outputs * (1-self.cumulative_outputs)
            i = np.transpose(self.cumulative_inputs)
            weight_gradient += i * (errors * regularizer)
        #hidden layer:
        else: #the output of the current layer is the input to another hidden layer
            if (len(self.cumulative_outputs[0]) >1): #this is a classifcation problem, and we want to multiply by the derivative of our sigmoid
                 regularizer = self.cumulative_outputs * (1-self.cumulative_outputs)
            else:
                 regularizer = 1
            # for i in range(len(self.weight_matricies[layer_no-1][0])):
            #     matrix = self.weight_matricies[layer_no-1]
            #     i = np.transpose(np.transpose(matrix)[i])
            #     i.shape = (len(i),1)
            #     regularizer = self.hidden_layers[layer_no -1] * (1-self.hidden_layers[layer_no -1])
            #     weight_gradient += i * (errors * regularizer)

            i = self.cumulative_hidden_layers[layer_no-1-1]
            print("errors ", errors.shape)
            print(regularizer.shape)
            print(i.shape)
            error_regularizer = errors * regularizer
            print(error_regularizer)
            weight_gradient += np.dot(np.transpose(i), error_regularizer)
            print("weight gradient ", weight_gradient.shape)
  

        self.weight_matricies[layer_no-1] -= self.learing_rate * weight_gradient
    """

    def backprop_peter(self):
        #update WMs
        i = len(self.weight_matricies)
        j = len(self.cumulative_hidden_layers)
        while i > 0:
            #this handles the case of no hidden layers
            if i == len(self.weight_matricies) and j == 0:
                if (len(self.cumulative_outputs[0]) >1): #this is a classifcation problem, and we want to multiply by the derivative of our sigmoid
                    #calculate regularizer with current output or layer nodes
                    regularizer = self.cumulative_outputs * (1 - self.cumulative_outputs)
                else:
                    #no regularizer for single output
                    regularizer = 1

                #calculate the change for the weight matrix
                delta_WM = self.delta_weight_matrix(self.weight_matricies[0], regularizer, self.errors, self.cumulative_inputs, self.previous_WM_delta[0])
                #claculate new weight matrix
                self.weight_matricies[0] -= delta_WM 
                #set previous delta for momentum use
                self.previous_WM_delta[0] = delta_WM
                break
            
            #handles initial backprop calc with hidden layers > 0
            elif i == len(self.weight_matricies) and j > 0:
                if (len(self.cumulative_outputs[0]) >1): #this is a classifcation problem, and we want to multiply by the derivative of our sigmoid
                    #calculate regularizer with current output or layer nodes
                    regularizer = self.cumulative_outputs * (1 - self.cumulative_outputs)
                else:
                    #no regularizer for single output
                    regularizer = 1
                #calculate the error for the next layer
                next_layer_error = np.dot(self.errors, np.transpose(self.weight_matricies[i-1]))  
                #calculate the change for the weight matrix
                delta_WM =  self.delta_weight_matrix(self.weight_matricies[i-1], regularizer, self.errors, self.cumulative_hidden_layers[j-1], self.previous_WM_delta[i-1])
                #claculate new weight matrix
                self.weight_matricies[i-1] -= delta_WM
                #set previous delta for momentum use
                self.previous_WM_delta[i-1] = delta_WM
            
            #handles backprop calc for rest of hidden layers when hidden layers > 0
            elif i < len(self.weight_matricies) and i > 1:
                #calculate the error for the next layer
                temp_next_layer_error = np.dot(next_layer_error, np.transpose(self.weight_matricies[i-1]))
                #calculate regularizer with current output or layer nodes
                regularizer = self.cumulative_hidden_layers[j] * (1 - self.cumulative_hidden_layers[j])
                #calculate the change for the weight matrix
                delta_WM = self.delta_weight_matrix(self.weight_matricies[i-1], regularizer, next_layer_error, self.cumulative_hidden_layers[j-1], self.previous_WM_delta[i-1])
                #claculate new weight matrix
                self.weight_matricies[i-1] -= delta_WM
                #set previous delta for momentum use
                self.previous_WM_delta[i-1] = delta_WM
                #set error for next layer
                next_layer_error = temp_next_layer_error
            
            #handles final backprop calc at final hidden layer
            else:
                #calculate regularizer with current output or layer nodes
                regularizer = self.cumulative_hidden_layers[0] * (1 - self.cumulative_hidden_layers[0])
                #calculate the change for the weight matrix
                delta_WM = self.delta_weight_matrix(self.weight_matricies[0], regularizer, next_layer_error, self.cumulative_inputs, self.previous_WM_delta[0])
                #claculate new weight matrix
                self.weight_matricies[0] +=  delta_WM
                #set previous delta for momentum use
                self.previous_WM_delta[0] = delta_WM

            #decrement counters
            i -= 1
            j -= 1
            #print("delta WM", delta_WM)

    #momentum can be used here
    def delta_weight_matrix(self, weight_matrix, regularizer, error_vector, avg_layer_values, previous_deltaWM):
        #uses equation delta_WM_i = learning_rate * Layer_Error_Vector * (avg_layer_output*(1-avg_layer_output)) * Avg_output_previous_layer_vector
        if (not self.momentum):
            error_regularizer = error_vector * regularizer
            delta_weight_matrix = self.learing_rate * np.dot(np.transpose(avg_layer_values), error_regularizer)

        # if momentum is being used
        elif (self.momentum):
            # print("Momentum is being used")
            error_regularizer = error_vector * regularizer
            delta_weight_matrix = self.learing_rate * np.dot(np.transpose(avg_layer_values), error_regularizer) + self.momentum_factor * previous_deltaWM

        else:
            raise Exception ("Specify momentun y or n")

        return delta_weight_matrix
    
    def classify_batch(self, test_data):
        tuples = []
        for i in test_data:
            point_class = i[0]
            guess = self.classify(i[1:])
            tuples.append([point_class,guess])
        #print(tuples)
        return tuples

    def classify(self, point):
        layer_target_num=1
        curr_layer = point
        for i in range(len(self.hidden_layers)+1):
            #output layer:
            if(len(self.hidden_layers) < layer_target_num): # the current layers output will be the classification/output layer
                next_layer = self.outputs   #target layer of curr_layer
                weights = self.weight_matricies[i]  #weights mapping from curr_layer to target_layer
                curr_layer = self.feed_forward_layer(curr_layer,next_layer,weights)
                if (len(self.outputs[0]) >1): #this is a classifcation problem, and we want our outputs to be between 0-1
                    curr_layer = self.sigmoidify_layer(curr_layer)
                self.outputs = curr_layer

            #hidden layer:
            else: #the output of the current layer is the input to another hidden layer
                next_layer = self.hidden_layers[layer_target_num-1]   #target layer of curr_layer
                weights = self.weight_matricies[i]  #weights mapping from curr_layer to target_layer
                curr_layer = self.feed_forward_layer(curr_layer,next_layer,weights)
                curr_layer = self.sigmoidify_layer(curr_layer)
                self.hidden_layers[layer_target_num-1] = curr_layer

            layer_target_num += 1 #iterate the target layer to next layer
        #print(self.outputs)
        return np.argmax(self.outputs[0])+1 #index of max (self.outputs)



    # TODO - F Score and Regression