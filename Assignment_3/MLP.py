import numpy as np

class MLP():
    #input-> #data_as_2dList, number_of_hidden_layers, number_of_hidden_nodes_in_each_layer
    def __init__(self, data, outputs, number_of_layers, number_of_nodes):
        self.data = data
        self.outputs = outputs #should this be a list of the possible classes????????
        
        self.hidden_layers = []
        for layer in range(number_of_layers):
            # self.hidden_layers.append(np.transpose(np.random.rand(number_of_nodes[layer])))
            self.hidden_layers.append(np.random.rand(1,number_of_nodes[layer]))
            

        # self.hidden_nodes = hidden_nodes
        
        if (len(number_of_nodes) != number_of_layers):
            raise Exception ("we need to know how many nodes are in each hidden layer")

        #make an list of np wieght matricies
        #weight_matricies = [np[hidden_nodes][weight_to_next_layer], np[]...  ]
        self.weight_matricies = []

        #number_of_nodes = [len(self.data[0])]+number_of_nodes +[outputs]
        
        if number_of_layers > 0:
            self.weight_matricies.append(np.random.rand(len(self.data[0]), number_of_nodes[0]))
            for i in range(number_of_layers-1):
                layer = np.random.rand(number_of_nodes[i], number_of_nodes[i+1])
                self.weight_matricies.append(layer)
            self.weight_matricies.append(np.random.rand(number_of_nodes[-1], outputs))
        else: 
            self.weight_matricies.append(np.random.rand(len(self.data[0]), outputs))

        # print(self.weight_matricies[2].shape)
        # print(self.hidden_layers[1].shape)
    
    def train(self):
        temp = 0
        while(temp!=self.matricies):
            temp = self.matricies

            
    
    
    # gradient:


    def sigmoidify_layer(self, layer):
        for i in range(len(layer)):
            layer[i] = sigmoid(layer[i])
        return layer

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x)) 

    def feed_forward(self):
        for input in self.data:
            values = np.transpose(input)
            for wm in range(len(weight_matricies)):
                #multiply weight matrix
                values = values * weight_matricies[wm]
                if wm <= len(hidden_layers):
                    #calculate node values of layer
                    raw_layer = values * hidden_nodes[wm]
                    #convert to sigmoid value
                    sigmoid_layer = sigmoidify_layer(raw_layer)
                    #save sigmoid values of layer
                    hidden_nodes[wm] = sigmoid_layer
                    values = sigmoid_layer

            #Calculate Errors
            #total

            #by class

    def network_train(self):
        #for every data point vector
        for d in self.data:

            curr_layer = d  #for any two adjacent layers, curr_layer is the input layer
            layer_target_num = 1 # the layer (not exclusively hidden) the local output is targeting
            #for every weight matrix (# of hidden layers + 1)
            for i in range(len(self.hidden_layers)+1):
                if(len(self.hidden_layers)> layer_target_num): # the current layers output will be the classification/output layer
                    pass
                else: #the output of the current layer is the input to another hidden layer
                    next_layer = self.hidden_layers[layer_target_num]   #target layer of curr_layer
                    weights = self.weight_matricies[i]  #weights mapping from curr_layer to target_layer
                    curr_layer = feed_forward_layer(curr_layer,next_layer,weights)
                    curr_layer = sigmoidify_layer(curr_layer)
                    self.hidden_layers[layer_target_num] = curr_layer


    
    def feed_forward_layer(self,layer1,layer2,layer1_weights):
        #self.hidden_layers
        #self.weight_matricies[np[][],np[][]...]
        for i in range(len(layer2)):
            #each node in the second layer is the dotproduct of the first layer and the first layer's weight functions
            layer2[i] = np.dot(np.transpose(layer1), layer1_weights[:,i])

        return layer2



    
   
    




