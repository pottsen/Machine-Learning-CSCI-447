import numpy as np



class MLP():
    #input-> #data_as_2dList, possible_outputs, number_of_hidden_layers, number_of_hidden_nodes_in_each_layer
    def __init__(self, data, output, number_of_layers, number_of_nodes):
        self.learing_rate = .1
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



        #number_of_nodes = [len(self.data[0])]+number_of_nodes +[outputs]

        #make an list of np wieght matricies
        #weight_matricies = [np[hidden_nodes][weight_to_next_layer], np[]...  ]
        self.weight_matricies = []
        if number_of_layers > 0:
            self.weight_matricies.append(np.random.rand(len(self.data[0])-1, number_of_nodes[0]))
            for i in range(number_of_layers-1):
                layer = np.random.rand(number_of_nodes[i], number_of_nodes[i+1])
                self.weight_matricies.append(layer)
            self.weight_matricies.append(np.random.rand(number_of_nodes[-1], len(output)))
        else:
            self.weight_matricies.append(np.random.rand(len(self.data[0])-1, len(output)))



        self.errors = np.zeros(len(output))
        self.error_partials = np.zeros(len(output))
        self.total_error = 0
        # print(self.weight_matricies[2].shape)
        # print(self.hidden_layers[1].shape)

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
        while(not equal):
            temp = []
            for i in self.weight_matricies:
                temp.append(np.copy(i))
            self.network_train_iteration()
            #print("outputs\n",self.outputs)
            print(self)
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
        self.errors = np.zeros(self.outputs.shape)
        #self.total_error = 0
        layer_target_num = 0
        #for every data point (vector)
        for d in self.data:

            actual = d[0] #first index is the class

            curr_layer = np.transpose(d[1:])  #for any two adjacent layers, curr_layer is the input layer

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
            #print("outputs")
            #print(self.outputs)

            #calculate errors Total and by class
            self.errors += self.error_update(actual, self.outputs)
        self.errors = self.errors / len(self.data)
        #print("ERROR:",str(self.errors))


        self.backpropagate(layer_target_num - 1, self.errors)


    #calculating node values for a hidden layer
    def feed_forward_layer(self,layer1,layer2,layer1_weights):
        #self.hidden_layers
        #self.weight_matricies[np[][],np[][]...]
        #for i in range(len(layer2)):
            #each node in the second layer is the dotproduct of the first layer and the first layer's weight functions
            # layer2[i] = np.dot(np.transpose(layer1), layer1_weights[:,i])
        #    layer2[i] = np.dot(layer1, layer1_weights[:,i])  #TODO: add bias here

        layer2 = np.dot(layer1,layer1_weights)
        layer2.shape = (1,len(layer1_weights[0]))
        return layer2

    #inputs-> actual class index,  output array values


    #input example-> 3, [.8,.2,.6]
    #actual is either the class index from 1-n or the value of the discrete class
    def error_update(self,actual, outputs): #(MSE)

        if len(outputs[0]) == 1:  #regression problem
            errors = [ (outputs[0] - actual)]#**2 ]
        else:  #classification
            if(type(actual) == int or type(actual) == float ):
                actual_vector = np.zeros((1,len(outputs[0])))
                actual_vector[0][int(actual) - 1] = 1    #ex: actual = 3 -->   [0, 0, 1]

            else:
                actual_vector = actual


            #errors = (guessed - actual)^2
            errors = np.subtract(outputs, actual_vector) #np array
            #print("errors\n",errors)
            #errors = np.power(errors, 2)

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


    def backpropagate(self, layer, errors):
        #for every weight matrix (# of hidden layers + 1)
        feed_back_values = np.transpose(self.outputs)
        for i in range(len(self.hidden_layers)+1):

            if(layer > 1): #we are at least past the first hidden layer, so we need to backpropogate the "actual" values of the previous hidden layer
                #reverse_weights = 1/self.weight_matricies[layer-1]
                print(feed_back_values.shape)
                reverse_weights = self.weight_matricies[layer-1]
                feed_back_values = np.dot(reverse_weights, feed_back_values)
                prev_error = self.error_update(np.transpose(feed_back_values), self.hidden_layers[layer-2])[0]

            self.backpropagate_layer(layer, errors)

            if (layer> 1):
                errors = prev_error


            layer -= 1 #iterate the target layer to next layer


    def backpropagate_layer(self,layer_no,errors):

        # ∇E(h,n) = 1/n * Σ (g_n - a_n) * regularizer' * weight_n_h
        # errors = [1/n * Σ (g_n0 - a_n0), 1/n * Σ (g_n1 - a_n1), ...]
        # regulizer is the derivative of the node normalization function
        # n is the output node
        # h is a hidden layer node
        # weight_n_h is the weight from node h to node n

        # partial_totalE_Wij = pE_t/pO_i * pO_i/pNet_i * pNet_i/pW_ijhttp://csci491-01.cs.montana.edu/~w32g348/www/montanahang/

        weight_gradient = np.zeros((self.weight_matricies[layer_no-1].shape))

        #output layer:
        if(len(self.hidden_layers) < layer_no): # the current layers output will be the classification/output layer
            if (len(self.outputs[0]) >1): #this is a classifcation problem, and we want to multiply by the derivative of our sigmoid
                regularizer = self.outputs * (1-self.outputs)
            else:
                regularizer = 1

            for i in range(len(self.weight_matricies[layer_no-1][0])):
                matrix = self.weight_matricies[layer_no-1]
                i = np.transpose(np.transpose(matrix)[i])
                i.shape = (len(i),1)
                weight_gradient += i * (errors * regularizer)

        #hidden layer:
        else: #the output of the current layer is the input to another hidden layer
            for i in range(len(self.weight_matricies[layer_no-1][0])):
                matrix = self.weight_matricies[layer_no-1]
                i = np.transpose(np.transpose(matrix)[i])
                i.shape = (len(i),1)
                regularizer = self.hidden_layers[layer_no -1] * (1-self.hidden_layers[layer_no -1])
                weight_gradient += i * (errors * regularizer)

        self.weight_matricies[layer_no-1] -= self.learing_rate * weight_gradient



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
        print(self.outputs)
        return self.outputs #index of max (self.outputs)


    #input-> error asscoiate with each output neron
    def backpropogate_layer_bruce_try(output_errors):

        #go through each hidden layer
            #hidden error for layer [i] = hidden layer[i] weights * error of layer[i+1]


        for i in range(len(output_errors)):
            #weight to output mode
            weights = self.weight_matricies[-1]
            for j in range(len(weights)):
                hidden_error = weights[j]/ np.sum(a=weights) * output_errors[i]
