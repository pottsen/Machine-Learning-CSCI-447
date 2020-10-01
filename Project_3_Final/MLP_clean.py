import numpy as np

class MLP():
    #input-> #data_as_2dList, possible_outputs, number_of_hidden_layers, number_of_hidden_nodes_in_each_layer, momentum(T/F)
    def __init__(self, data, output, number_of_layers, number_of_nodes, momentum):
         
        ### Initialize variables
        # flag
        self.momentum = bool(momentum)
        # learning rates
        self.momentum_factor = 0.1
        self.learing_rate = 0.05

        # for use in making data structures
        self.output_size = output
        self.data = data
        self.number_of_layers = 2 + int(number_of_layers)
        self.number_of_weight_matrices = self.number_of_layers -1

        # tracking vairables
        self.outputs = np.zeros(len(output))
        self.outputs.shape = (1,len(output))

        # check for correct inputs
        if (len(number_of_nodes) != number_of_layers):
            raise Exception ("we need to know how many nodes are in each hidden layer")

        #make a layers list of np arrays. input is first layer and output is last layer
        self.layers = []
        for layer in range(int(self.number_of_layers)):
            if layer == 0:
                self.layers.append(np.zeros((len(data[0])-1,1)))
            elif layer == int(self.number_of_layers)-1:
                self.layers.append(np.zeros((len(output), 1)))
            else:
                self.layers.append(np.zeros((number_of_nodes[layer-1], 1)))
            print("Layer Shape\n", self.layers[layer].shape)

        #make an list of wieght matricies and previous deltas for momentum
        self.weight_matricies = []
        self.previous_WM_delta = []
        for i in range(int(self.number_of_layers)-1):
            x ,y = self.layers[i].shape
            x2 ,y2 = self.layers[i+1].shape
            self.weight_matricies.append(np.random.rand(x2, x))
            self.previous_WM_delta.append(np.zeros((x2, x)))

    def train(self):
        temp = []
        equal = False
        iterations = 0
        while(not equal) and iterations < 1000:
            # store weight matrices to check for convergence
            temp = []
            for i in self.weight_matricies:
                temp.append(np.copy(i))

            self.network_train_iteration()
            # print("outputs\n",self.outputs)
            # print(self)
            iterations +=1
            print(iterations)
            equal = True
            #check for convergence
            for i in range(len(self.weight_matricies)):
                if(not np.array_equal(self.weight_matricies[i].round(decimals = 4),temp[i])):
                    equal = False
            # print(equal)

    def network_train_iteration(self):
        # create variable to track error
        self.errors = np.zeros(self.layers[-1].shape)

        #create target vector to compare to output
        if len(self.output_size) > 1:
            self.target_vector = np.zeros((np.shape(self.layers[-1])))
        else: 
            self.target_vector = np.ones(1)
        
        # itereate through data and train
        for d in self.data:
            #reset target vector for each data point
            self.target_vector = np.multiply(self.target_vector,0)
            target_class = d[0]
            if len(self.output_size) > 1:
                self.target_vector[(int(target_class)-1)] = 1
            else: 
                self.target_vector = target_class

            #set inputs
            inputs = np.vstack(d[1:])

            # feed data point through network
            self.feed_forward(inputs)
            
            #calculate error on the output
            self.error = np.subtract(self.layers[-1], self.target_vector)

            # backpropogate error to train
            self.back_propogation_single()
            
    def feed_forward(self, inputs):
        self.layers[0] = inputs

        for i in range(len(self.weight_matricies)):
            #check if regression problem so we dont sigmodify the final output
            if self.layers[-1].shape == self.layers[i+1].shape and len(self.output_size) == 1:
                self.layers[i+1] = np.dot(self.weight_matricies[i], self.layers[i])
            else:
                self.layers[i+1] = np.dot(self.weight_matricies[i], self.layers[i])
                #sigmoid calculation
                exp = np.exp(-self.layers[i+1])
                exp_1 = np.add(exp, 1)
                self.layers[i+1] = 1/exp_1
        # print("output", self.layers[-1])
        return self.layers[-1]

    def back_propogation_single(self):
        i = len(self.weight_matricies)
        #initialize next_error to current error
        next_error = self.error
        while i > 0: #iterate back through layers
            regularizer = np.multiply(self.layers[i], np.subtract(1,self.layers[i]))
            regularizer_error = np.multiply(next_error, regularizer)
            delta_WM = self.learing_rate * np.dot(regularizer_error, np.transpose(self.layers[i-1]))
            next_error = np.dot(np.transpose(self.weight_matricies[i-1]), next_error)
            if self.momentum == True:
                delta_WM_momentum = delta_WM + np.multiply(self.previous_WM_delta[i-1],self.momentum_factor)
                self.previous_WM_delta[i-1] = delta_WM
                self.weight_matricies[i-1] -= delta_WM_momentum
                
            else:
                self.weight_matricies[i-1] -= delta_WM

            i -= 1

    def __str__(self):
        stringify = "NETWORK:"
        stringify+= "\nINPUT:\n"+str(self.layers[0])
        for i in range(len(self.weight_matricies)):
            stringify+= "\nWEIGHTS:\n"+str(self.weight_matricies[i])
        stringify += "\nOUTPUT\n"+str(self.layers[-1])
        stringify += "\nEND NETWORK"
        return stringify

    def classify(self, test_data):
        #run test data and return combination of predicted and target/actual class
        tuples = []
        for i in test_data:
            target = np.array(i[0])
            inputs = np.vstack(i[1:])
            outputs = self.feed_forward(inputs)
            guess = np.argmax(outputs)
            tuples.append([target, guess])
        return tuples