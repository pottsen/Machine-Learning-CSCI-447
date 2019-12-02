from mlp import MLP
import random
import copy

class PopulationManager():

    def __init__(self, pop_size, mlp_dims):
        self.population = []
        for i in range(pop_size):
            self.population.append(MLP(mlp_dims))

    def uniform_cross(self, parent1, parent2):  #parents are MLPs
        child = copy.deepcopy(parent1)

        parent1 = parent1.unzip_neuron()
        parent2 = parent2.unzip_neuron()

        child_dna = []
        for i in range(len(parent1)):
            coin = random.randrange(0,2,1)
            if(coin == 0):
                child_dna.append(parent1[i])
            else:
                child_dna.append(parent2[i])

        child.rezip_neuron(child_dna)

        return child
