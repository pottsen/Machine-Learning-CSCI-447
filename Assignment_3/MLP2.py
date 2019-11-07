import numpy as np

class MLP2():
    #input-> #data_as_2dList, possible_outputs, number_of_hidden_layers, number_of_hidden_nodes_in_each_layer
    def __init__(self, data, output, number_of_layers, number_of_nodes, momentum):
        # flag 
        self.momentum = bool(momentum)
        self.momentum_factor = 0.3
        self.learing_rate = 0.1
        self.output_size = output
        self.data = data
        self.number_of_layers = 2 + int(number_of_layers)
        self.number_of_weight_matrices = self.number_of_layers -1
       
        self.outputs = np.zeros(len(output))
        self.outputs.shape = (1,len(output))


        if (len(number_of_nodes) != number_of_layers):
                raise Exception ("we need to know how many nodes are in each hidden layer")

        #make a layers list of np arrays
        self.layers = []
        for layer in range(int(self.number_of_layers)):
            if layer == 0:
                self.layers.append(np.zeros((len(data[0])-1,1)))
            elif layer == int(self.number_of_layers)-1:
                self.layers.append(np.zeros((len(output), 1)))
            else:
                self.layers.append(np.zeros((number_of_nodes[layer-1], 1)))
            print("Layer Shape\n", self.layers[layer].shape)

        print("len output\n", output)
        x,y = self.layers[0].shape
        print("Layer 0 len\n", y)

        #make an list of np wieght matricies and previous deltas for momentum
        self.weight_matricies = []
        self.previous_WM_delta = []
        for i in range(int(self.number_of_layers)-1):
            x ,y = self.layers[i].shape
            x2 ,y2 = self.layers[i+1].shape
            self.weight_matricies.append(np.random.rand(x2, x))
            self.previous_WM_delta.append(np.zeros((x2, x)))
        
        for i in range(len(self.weight_matricies)):
            print("layer shape\n", self.layers[i].shape)
            print("weight matices\n", self.weight_matricies[i].shape)

        # print("prev weight matices\n", self.previous_WM_delta)

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
                # self.weight_matricies[i] =self.weight_matricies[i].round(decimals = 4)
                # if(not np.array_equal(self.weight_matricies[i],temp[i])):
                #     equal = False
                if(not np.array_equal(self.weight_matricies[i].round(decimals = 4),temp[i])):
                    equal = False
            # print(equal)

    def network_train_iteration(self):
        self.errors = np.zeros(self.layers[-1].shape)
        # print("error shape", self.errors.shape)
        self.cumulative_layers = []
        for layer in range(len(self.layers)):
            self.cumulative_layers.append(np.zeros(self.layers[layer].shape))
        # print(self.cumulative_layers)
        if len(self.output_size) > 1:
            self.target_vector = np.zeros((np.shape(self.layers[-1])))
        else: self.target_vector = np.ones(1)
        # print("target shape\n", self.target_vector.shape)
        
        for d in self.data:
            # print(d)
            self.target_vector = np.multiply(self.target_vector,0)
            target_class = d[0]
            if len(self.output_size) > 1:
                self.target_vector[(int(target_class)-1)] = 1
            else: 
                self.target_vector = target_class

            #set inputs
            inputs = np.vstack(d[1:])
            # print("inputs", inputs)

            self.feed_forward(inputs)
            # print("output vector\n", self.layers[-1], self.layers[-1].shape)
            
            self.error = np.subtract(self.layers[-1], self.target_vector)
            # print("error\n", self.error, self.error.shape)
            self.back_propogation_single()
            
    def feed_forward(self, inputs):
        self.layers[0] = inputs
        # print("first layer", self.layers[0])
        for i in range(len(self.weight_matricies)):
            if self.layers[-1].shape == self.layers[i+1].shape and len(self.output_size) == 1:
                self.layers[i+1] = np.dot(self.weight_matricies[i], self.layers[i])
                # print(self.layers[i+1].shape)
            else:
                # print("wm shape i\n", self.weight_matricies[i].shape)
                # print("Layer shape before\n", self.layers[i].shape)
                # print("weight matrix\n", self.weight_matricies[i].shape)
                self.layers[i+1] = np.dot(self.weight_matricies[i], self.layers[i])
                # print("Layer shape after\n", self.layers[i+1].shape)
                #convert with sigmoid
                exp = np.exp(-self.layers[i+1])
                # print("exp\n", exp)
                exp_1 = np.add(exp, 1)
                # print("exp_1\n", exp_1)
                self.layers[i+1] = 1/exp_1
                # print("layer i+1\n",self.layers[i+1])
        return self.layers[-1]

    def back_propogation_single(self):
        i = len(self.weight_matricies)
        next_error = self.error
        while i > 0:
            # if self.momentum == True:
            # print("Layer shape\n", self.layers[i].shape)
            regularizer = np.multiply(self.layers[i], np.subtract(1,self.layers[i]))
            # print("regularizer\n", regularizer, regularizer.shape)
            regularizer_error = np.multiply(next_error, regularizer)
            # print("reularizer error\n", regularizer_error, regularizer_error.shape)
            # print("next layer\n", self.layers[i-1], self.layers[i-1].shape)
            delta_WM = self.learing_rate * np.dot(regularizer_error, np.transpose(self.layers[i-1]))
            # print("delta_wm\n", delta_WM)
            next_error = np.dot(np.transpose(self.weight_matricies[i-1]), next_error)
            if self.momentum == True:
                # print("Momemtum")
                delta_WM_momentum = delta_WM + np.multiply(self.previous_WM_delta[i-1],self.momentum_factor)
                self.previous_WM_delta[i-1] = delta_WM
                self.weight_matricies[i-1] -= delta_WM_momentum
            else:
                self.weight_matricies[i-1] -= delta_WM
            i -= 1

    # def sigmoidify_layer(self, layer):
    #     for i in range(len(layer)):
    #         layer[i] = self.sigmoid(layer[i])
    #     return layer

    # def sigmoid(self, x):
    #     return 1/(1 + np.exp(-x))

    def __str__(self):
        stringify = "NETWORK:"
        stringify+= "\nINPUT:\n"+str(self.layers[0])
        for i in range(len(self.weight_matricies)):
            stringify+= "\nWEIGHTS:\n"+str(self.weight_matricies[i])
            # stringify += "\nLAYER\n"+str(self.layers[i+1])
        stringify += "\nEND NETWORK"
        return stringify

    def classify(self, test_data):
        tuples = []
        for i in test_data:
            target = np.array(i[0])
            inputs = np.vstack(i[1:])
            outputs = self.feed_forward(inputs)
            guess = np.argmax(outputs)
            tuples.append([target, guess])

        return tuples