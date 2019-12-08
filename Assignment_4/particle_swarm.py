from population_manager import PopulationManager
import random
import numpy as np
import copy
from data_processing import Data_Processing

class particle_swarm(PopulationManager):

    def __init__(self, pop_size, mlp_dims, pBest_coeff, gBest_coeff, inertia_coeff, training_data, test_data):

        PopulationManager.__init__(self, pop_size, mlp_dims)
        self.pBest_C = pBest_coeff
        self.gBest_C = gBest_coeff
        self.pBest_fitness = [1000000000000]*pop_size
        self.pBest = self.population
        self.prev_velocity = self.population

        self.gBest_fitness = 1000000000000
        self.gBest = self.population[0]
        self.inertia_C = inertia_coeff
        # self.training_data = training_data
        # self.test_data = test_data
        self.count = 0

        self.training_data_outputs = []
        self.training_data_inputs = []
        for i in training_data:
            actual_class = i[0]
            temp = [0] * 29
            temp[int(actual_class-1)] = 1
            self.training_data_inputs.append(i[1:])
            self.training_data_outputs.append(temp)

        self.test_data_outputs = []
        self.test_data_inputs = []
        for i in test_data:
            actual_class = i[0]
            temp = [0] * 29
            temp[int(actual_class-1)] = 1
            self.test_data_inputs.append(i[1:])
            self.test_data_outputs.append(temp)


        PopulationManager.__init__(self, pop_size, mlp_dims)


    #new WM = old WM + velocity
    def velocity_calc(self, prev_velocity, weights, pBest_weights, gBest_weights):
        #weights = self.population[i].layers[j].next_weights
        velocity = []
        for i in range(len(prev_velocity)):
            v = self.inertia_C*prev_velocity[i] + self.pBest_C * random.uniform(0,1) * (pBest_weights[i] - weights[i]) + self.gBest_C * random.uniform(0,1) * (gBest_weights[i] - weights[i])
            velocity.append(v)
        return velocity

    def update(self):
        
        gBest_weights = self.gBest.unzip_neuron()

        for i in range(len(self.population)):
            print(i)
            weights = self.population[i].unzip_neuron()
            pBest_weights = self.pBest[i].unzip_neuron()
            if self.count == 0:
                prev_velocity = self.prev_velocity[i].unzip_neuron()
            else:
                prev_velocity = self.prev_velocity[i].unzip_neuron()
            velocity = self.velocity_calc(prev_velocity, weights, pBest_weights, gBest_weights)
            self.prev_velocity[i].rezip_neuron(velocity) 

            new_weights = weights + velocity
            self.population[i].rezip_neuron(new_weights)
            pBest_weights = None
            # self.pBest[i].rezip_neuron(pBest_weights)

        self.count +=1

    def run_PSO(self):
        #iteration count
        iteration = 0
        # if self.population[0].layers[-1].get_layer_size() > 1:
        #     output_array = np.zeros((len(self.trai),self.population[0].layers[-1].get_layer_size()))
        #     for i in range(len(outputs)):
        #         output_array[i][int(outputs[i])-1] = 1
        #calculate fitness
        while iteration < 100:
            print("iteration ", iteration)
            for i in range(len(self.population)):
                fitness = self.population[i].fitness(self.training_data_inputs, self.training_data_outputs)
                print("Fitness ", fitness)
                #if fitness < pBest reset
                if fitness < self.pBest_fitness[i]:
                    self.pBest[i] = self.population[i]
                    self.pBest_fitness[i] = fitness 
                #if fitness < gbest reset
                if fitness < self.gBest_fitness:
                    self.gBest = self.population[i]
                    self.gBest_fitness = fitness
                self.update() 
            iteration += 1

if __name__ == "__main__":

    #Prepping data ----------------------------
    data_aba = Data_Processing(["abalone",], [8], {"M":"1", "F":"2", "I":"3"})

    data_aba.load_data("./processed")

    #triming it down to 10
    data_aba.file_array['abalone'] = data_aba.file_array['abalone'][:100]
    print(data_aba.file_array['abalone'][0])
    #slice in to 5
    data_aba.slicer(5, "abalone")

    training_data = data_aba.file_array[0]
    test_data = data_aba.combine(data_aba.file_array[1:])
    #end --------------------------------------

    #tesing alogorithm ------------------------
    
    # print(len(data_aba.file_array[0][0][1:]))
    pso = particle_swarm(5, [(len(data_aba.file_array[0][0][1:])),30,29], 0.3, 0.3, 0.1, training_data, test_data)

    pso.run_PSO()

    #end --------------------------------------