import numpy as np

class MLP():
    #input-> #data_as_2dList, possible_outputs, number_of_hidden_layers, number_of_hidden_nodes_in_each_layer
    def __init__(self, data, output, number_of_layers, number_of_nodes):
        self.data = data
        self.outputs = np.zeros(len(output)) 
        self.outputs.shape = (len(output),1)
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
            self.weight_matricies.append(np.random.rand(len(self.data[0]), len(output)))



        self.errors = np.zeros(len(output))
        self.error_partials = np.zeros(len(output))
        self.total_error = 0
        # print(self.weight_matricies[2].shape)
        # print(self.hidden_layers[1].shape)
    
    def train(self):
        temp = 0
        while(temp!=self.weight_matricies):
            temp = self.weight_matricies
            self.network_train_iteration()
  
    # gradient:


    def sigmoidify_layer(self, layer):
        for i in range(len(layer)):
            layer[i] = self.sigmoid(layer[i])
        return layer

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x)) 


    def network_train_iteration(self):
        self.errors = 0
        self.total_error = 0
        
        #for every data point (vector)
        for d in self.data:

            actual = d[0] #first index is the class

            curr_layer = d[1:]  #for any two adjacent layers, curr_layer is the input layer

            layer_target_num = 1 # the layer (not exclusively hidden) the local output is targeting
            
            #for every weight matrix (# of hidden layers + 1)
            for i in range(len(self.hidden_layers)+1):
                #output layer:
                if(len(self.hidden_layers) < layer_target_num): # the current layers output will be the classification/output layer
                    next_layer = self.outputs   #target layer of curr_layer
                    weights = self.weight_matricies[i]  #weights mapping from curr_layer to target_layer
                    curr_layer = self.feed_forward_layer(curr_layer,next_layer,weights)
                    if (len(self.outputs) >1): #this is a classifcation problem, and we want our outputs to be between 0-1
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
            
            
        #backprop


    #calculating node values for a hidden layer
    def feed_forward_layer(self,layer1,layer2,layer1_weights):
        #self.hidden_layers
        #self.weight_matricies[np[][],np[][]...]
        for i in range(len(layer2)):
            #each node in the second layer is the dotproduct of the first layer and the first layer's weight functions
            # layer2[i] = np.dot(np.transpose(layer1), layer1_weights[:,i])
            layer2[i] = np.dot(layer1, layer1_weights[:,i])  #TODO: add bias here
        return layer2

    #inputs-> actual class index,  output array values

    
    #input example-> 3, [.8,.2,.6]
    #actual is either the class index from 1-n or the value of the discrete class
    def error_update(self,actual, outputs): #(MSE)

        if len(outputs) == 1:  #regression problem
            errors = [ (outputs[0] - actual)**2 ]
        else:  #classification
            actual_vector = np.zeros((len(outputs),1)) 
            actual_vector[int(actual) - 1] = 1    #ex: actual = 3 -->   [0, 0, 1]
            

            #errors = (guessed - actual)^2
            errors = np.subtract(outputs, actual_vector) #np array
            errors = np.power(errors, 2)
            
            
        return errors

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

    def backpropogate_layer():
        pass
        # ∇E = 1/n * Σ (g_n - a_n) * sigmoid ** -1 * actual
        # partial_totalE_Wij = pE_t/pO_i * pO_i/pNet_i * pNet_i/pW_ijhttp://csci491-01.cs.montana.edu/~w32g348/www/montanahang/





    #input-> error asscoiate with each output neron
    def backpropogate_layer_bruce_try(output_errors):
        
        #go through each hidden layer
            #hidden error for layer [i] = hidden layer[i] weights * error of layer[i+1]


        for i in range(len(output_errors)):
            #weight to output mode
            weights = self.weight_matricies[-1]
            for j in range(len(weights)):
                hidden_error = weights[j]/ np.sum(a=weights) * output_errors[i]

