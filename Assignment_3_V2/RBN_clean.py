import numpy as np

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
        # find max distance for use in stdev calc
        self.d_max = self.distance_max(self.centers)
        #calculate stdev of hidden nodes
        self.stdev = float(self.d_max) / ((2 * len(self.centers))**0.5)
        self.errors = np.zeros(self.outputs.shape)

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
        while(not equal) and iterations < 2:
            # store weight matrices to check for convergence
            tempWM = np.copy(self.weight_matrix)
            self.network_train_iteration()
            print(self)
            iterations +=1
            print("Iteration ", iterations)
            equal = True

            # check for convergence
            if(not np.array_equal(self.weight_matrix, tempWM)):
                    equal = False

    def network_train_iteration(self):            
        #for every data point (vector)
        for d in self.data[:2]:
            self.target = d[0] #first index is the class
            inputs = np.stack(d[1:])
            centers_no_class = self.centers[:,1:]

            # run data point through network
            self.feed_forward(inputs, centers_no_class)

            # calculate errors
            if len(self.outputs) == 1:  #regression problem
                self.errors = (self.target - self.outputs)

            else:  #classification
                self.target_vector = np.zeros((self.outputs.shape))
                self.target_vector[(int(self.target)-1)] = 1 #ex: target = 3 -->   [0, 0, 1]
                self.errors = np.subtract(self.target_vector, self.outputs) #np array

            #backpropogate error through and train
            self.backprop()


    def feed_forward(self, inputs, centers_no_class):
        #### Calculate node values
        # calculate attribute distances from centroids/nodes
        deltas = np.subtract(inputs, centers_no_class)
        deltas_squared = np.square(deltas)
        exp_inside = np.divide(deltas_squared, (-2 * self.stdev**2))
        sum_exp_inside = np.zeros((len(exp_inside),1))
        for i in range(len(exp_inside)):
            sum_exp_inside[i,0] = np.sum(exp_inside[i])
        exp = np.exp(sum_exp_inside)
        self.rbf_layer = exp

        #now multiply RBF layer by weights to get output
        self.outputs = np.dot(self.weight_matrix, self.rbf_layer)
        self.outputs = np.divide(self.outputs, self.number_of_nodes)
        
    ###calculates max distance between the hidden layer nodes for used in stdev calc
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

    def backprop(self):
        #update WM
        delta_WM = self.learning_rate * np.dot(self.errors, np.transpose(self.rbf_layer))
        self.weight_matrix += delta_WM


    #takes a list of training points and returns a list of tuples corresponding to them (actual class/value, guessed class/value)
    ## for use in F-Score and Regession calculations
    def classify(self, test_data):
        centers_no_class = self.centers[:,1:]
        center_classes = self.centers[:,0]
        guesses = []
        for i in test_data:
            target = i[0]
            data = i[1:]
            self.feed_forward(data, centers_no_class)
            # output = self.outputs
            if(len(self.outputs)==1):#regression
                output = self.outputs
            else:#classification
                output = np.argmax(self.outputs[0])+1 #index of max (self.outputs)
    
            guesses.append([target,output])
        return guesses
        