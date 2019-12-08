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
        self.pBest_fitness = [0]*pop_size
        self.pBest = self.population
        self.prev_velocity = self.population
        print("pop len", len(self.population))
        self.gBest_fitness = 0
        self.gBest = self.population[0]
        self.inertia_C = inertia_coeff
        self.count = 0

        self.training_data_outputs = []
        self.training_data_inputs = []
        for i in training_data:
            actual_class = i[0]
            temp = [0] * mlp_dims[-1]
            temp[int(actual_class-1)] = 1
            self.training_data_inputs.append(i[1:])
            self.training_data_outputs.append(temp)

        self.test_data_outputs = []
        self.test_data_inputs = []
        for i in test_data:
            actual_class = i[0]
            temp = [0] * mlp_dims[-1]
            temp[int(actual_class-1)] = 1
            self.test_data_inputs.append(i[1:])
            self.test_data_outputs.append(temp)


        PopulationManager.__init__(self, pop_size, mlp_dims)


    #new WM = old WM + velocity
    def velocity_calc(self, prev_velocity, weights, pBest_weights, gBest_weights):
        #weights = self.population[i].layers[j].next_weights
        velocity = []
        for i in range(len(prev_velocity)):
            v = self.inertia_C*prev_velocity[i] + self.pBest_C*random.uniform(0,1) * (pBest_weights[i] - weights[i]) + self.gBest_C * random.uniform(0,1) * (gBest_weights[i] - weights[i])
            # print(v)
            velocity.append(v)
        return velocity

    def update(self):
        
        gBest_weights = self.gBest.unzip_neuron()

        for i in range(len(self.population)):
            # print(i)
            old_pop = self.population[i]
            weights = self.population[i].unzip_neuron()
            pBest_weights = self.pBest[i].unzip_neuron()
            prev_velocity = self.prev_velocity[i].unzip_neuron()

            velocity = self.velocity_calc(prev_velocity, weights, pBest_weights, gBest_weights) 

            new_weights = []
            for j in range(len(velocity)):
                # print("i",i)
                nw= weights[j] + velocity[j]
                new_weights.append(nw)

            # print("velocity\n", velocity[:20])
            print("New equal old ", new_weights == weights)
            # print("pv len", len(self.prev_velocity))
            self.prev_velocity[i].rezip_neuron(velocity)
            self.pBest[i].rezip_neuron(pBest_weights)

            self.population[i].rezip_neuron(new_weights)
            print("pop same ", old_pop == self.population[i])

        self.gBest.rezip_neuron(gBest_weights)
        self.count +=1

    def run_PSO(self):
        #iteration count
        iteration = 0
        #calculate fitness
        while iteration < 10:
            # print("iteration ", iteration)
            print("gBest Fitness ", self.gBest_fitness)
            for i in range(len(self.population)):
                fitness = self.population[i].fitness(self.training_data_inputs, self.training_data_outputs)
                # print("Fitness ", fitness)
                #if fitness < pBest reset
                if fitness > self.pBest_fitness[i]:
                    print("pBest updated")
                    self.pBest[i] = self.population[i]
                    self.pBest_fitness[i] = fitness 
                #if fitness < gbest reset
                if fitness > self.gBest_fitness:
                    print("gBest updated")
                    self.gBest = self.population[i]
                    self.gBest_fitness = fitness

            self.update() 
            iteration += 1
            print("iteration ", iteration, " fitness ", self.gBest_fitness)

if __name__ == "__main__":

    #Prepping data ----------------------------
    data_aba = Data_Processing(["abalone",], [8], {"M":"1", "F":"2", "I":"3"})

    data_aba.load_data("./processed")

    #triming it down to 10
    data_aba.file_array['abalone'] = data_aba.file_array['abalone'][:5]
    #slice in to 5
    data_aba.slicer(5, "abalone")

    test_data = data_aba.file_array[0]
    training_data = data_aba.combine(data_aba.file_array[1:])
    #end --------------------------------------

    #tesing alogorithm ------------------------
    
    # print(len(data_aba.file_array[0][0][1:]))
    pso = particle_swarm(1, [(len(data_aba.file_array[0][0][1:])),29], 2, 2, 0.75, training_data, test_data)

    pso.run_PSO()

    #end --------------------------------------