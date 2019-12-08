from mlp import MLP
import random
import copy

class PopulationManager():

    def __init__(self, pop_size, mlp_dims):
        self.mutation_rate = 0.1
        self.mutation_value = 0.1
        self.population = []
        for i in range(pop_size):
            self.population.append(MLP(mlp_dims))

    def main(self):  #this will controll the population over the generations
        #for x number of generations, preform generation
        pass

    def generation(self):
        #ass fitness
        #crossover
        #mutate
        pass

    def selection(self):
        pass

    def uniform_cross(self, parent1, parent2):  #parents are MLPs
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)

        parent1 = parent1.unzip_neuron()
        parent2 = parent2.unzip_neuron()

        child1_dna = []
        child2_dna = []
        for i in range(len(parent1)):
            coin = random.randrange(0,2,1)
            if(coin == 0):
                child1_dna.append(parent1[i])
                child2_dna.append(parent2[i])
            else:
                child1_dna.append(parent2[i])
                child2_dna.append(parent1[i])

        child1.rezip_neuron(child1_dna)
        child2.rezip_neuron(child2_dna)

        return child1, child2

    def mutation(self, mlp):
        gene = mlp.unzip_neuron()
        for i in range(len(gene)):
            coin = random.randrange(0,int(1/self.mutation_rate))
            if(coin == 0):
                gene[i] += gene[i]*self.mutation_value
        return gene
            
