from population_manager import PopulationManager
import random
import numpy as np

class particle_swarm(PopulationManager):

    def __init__(self, pop_size, mlp_dims, pBest_coeff, gBest_coeff, inertia_coeff, training_data, test_data):

        self.pBest_C = pBest_coeff
        self.gBest_C = gBest_coeff
        self.pBest = np.array(pop_size)*1000000000000
        self.gBest = 1000000000000
        self.inertia_C = inertia_coeff
        self.training_data = training_data
        self.test_data = test_data
        self.count = 0


        PopulationManager.__init__(self, pop_size, mlp_dims)


    #new WM = old WM + velocity
    def velocity_calc(self, prev_velocity, weights):
        #weights = self.population[i].layers[j].next_weights

        velocity = self.inertia_C*self.prev_velocity + self.pBest_C * random.random() * (self.pBest - weights) + self.gBest_C * random.random() * (self.gBest - weights)
        return velocity

    def update(self):
        # initialize previous velocity 
        # self.prev_velocity
        for i in self.population:
            for j in range(len(self.population[i].layers)-1):
                weights = self.population[i].layers[j].next_weights
                if count == 0:
                    self.prev_velocity = np.zeros(weights)
                velocity = self.velocity_calc(self.prev_velocity, weights)
                self.population[i].layers[j].next_weights = weights + velocity
                count +=1

    def run_PSO(self):
        #iteration count
        iteration = 0
        inputs = self.test_data[:][1:]
        outputs = self.test_data[:][0]
        if len(self.population[0].layers[-1]) > 1:
            output_array = np.zeros(len(outputs),len(self.population[0].layers[-1]))
            for i in outputs:
                output_array[i][outputs[i]-1] += 1
        #calculate fitness
        while iteration < 1000:
            for i in self.population:
                fitness = self.population[i].fitness(inputs, outputs)
                #if fitness < pBest reset
                if fitness < self.pBest[i]:
                    self.pBest[i] = fitness 
                #if fitness < gbest reset
                if fitness < self.gBest:
                    self.gBest = fitness 

    