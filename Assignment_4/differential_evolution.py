from population_manager import PopulationManager
import random
import numpy as np
import copy
from data_processing import Data_Processing

class Differential_Evolution(PopulationManager):

    def __init__(self, pop_size, mlp_dims, d_weight, training_data, test_data):
        self.mlp_dims = mlp_dims
        self.d_weight = d_weight  #between 0 and 2
        self.training_data = training_data
        self.test_data = test_data

        PopulationManager.__init__(self, pop_size, mlp_dims)
        self.best = self.population[0]
        self.best_f = 0

    def mutation(self):
        pop_size = len(self.population)
        for i in range(pop_size):
            members = []

            #pick indexes of 3 unique random members of population
            while(len(members)<3):
                rando = random.randrange(0,pop_size)
                if(rando not in members):
                    members.append(rando)

            new_member = copy.deepcopy(self.population[i])
            a = self.population[members[0]].unzip_neuron()
            b = self.population[members[1]].unzip_neuron()
            c = self.population[members[2]].unzip_neuron()

            y = []
            for dim in range(len(a)):
                y_i = a[dim] + self.d_weight * (b[dim] - c[dim])
                y.append(y_i)

            new_member.rezip_neuron(y)

            new_member, throwaway = self.uniform_cross(self.population[i],new_member)

            classes = []
            dat = []
            l1 = copy.deepcopy(self.training_data)
            random.shuffle(self.training_data)

            for d in self.training_data[:100]:
                c1 = [0]* self.mlp_dims[-1]
                c1[int(d[0])] = 1
                classes.append(c1)
                dat.append(d[1:])
            self.training_data = l1

            f1 = self.population[i].fitness(dat,classes)
            f2 = new_member.fitness(dat,classes)

            if(f2>f1):
                self.population[i] = new_member

            if(f2>self.best_f):
                self.best = self.population[i]
                self.best_f = f2
                print(f2)

if __name__ == "__main__":

    data_aba = Data_Processing(["abalone",], [8], {"M":"1", "F":"2", "I":"3"})
    data_aba.load_data("./processed")


    #slice in to 5
    data_aba.slicer(5, "abalone")

    test_data = data_aba.file_array[0]
    training_data = data_aba.combine(data_aba.file_array[1:])


    de = Differential_Evolution(100,[8,15,29],1.5,training_data,test_data)
    for i in range(100):
        de.mutation()

    best = de.best
    for n in range(20):
        print(test_data[n])
        b = 0
        bidx = 0
        for i in range(len(best.predict(test_data[n][1:]))):
            if(best.predict(test_data[n][1:])[i]>b):
                b = best.predict(test_data[n][1:])[i]
                bidx = i
        print(bidx)
