from population_manager import PopulationManager
import random
import numpy as np
import copy
from data_processing import Data_Processing
from evaluations import  f_score, mse

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

            if(self.mlp_dims[-1]==1):
                for d in self.training_data[:100]:
                    c1 = [0]* self.mlp_dims[-1]
                    classes.append(d[0])
                    dat.append(d[1:])
            else:
                for d in self.training_data[:100]:
                    c1 = [0]* self.mlp_dims[-1]
                    c1[int(d[0])] = 1
                    classes.append(c1)
                    dat.append(d[1:])

            self.training_data = l1

            f1 = self.population[i].fitness_with_f1(dat,classes)
            f2 = new_member.fitness_with_f1(dat,classes)

            if(f2>f1):
                self.population[i] = new_member

            if(f2>self.best_f):
                self.best = self.population[i]
                self.best_f = f2
                print(f2)

    def run(self, generation_limit):
        for i in range(generation_limit):
            de.mutation()
        return self.best

    def metrics(self):
        guesses = []
        if(self.mlp_dims[-1]==1): #regression
            for n in range(len(self.test_data)):
                guess = self.best.predict(self.test_data[n][1:])[0]
                actual = self.test_data[n][0]
                guesses.append([actual,guess])
            score = {"MSE": mse(guesses)}



        else:   #classification
            for n in range(len(self.test_data)):
                b = 0
                guess = 0
                for i in range(len(self.best.predict(self.test_data[n][1:]))):
                    if(self.best.predict(self.test_data[n][1:])[i]>b):
                        b = self.best.predict(self.test_data[n][1:])[i]
                        bidx = i
                actual = self.test_data[n][0]

                guesses.append([actual,guess])
            score = f_score(guesses)
        #print(score)
        #print(guesses[0])
        return score, guesses


if __name__ == "__main__":

    sets = ["abalone","car","forestfires","machine","segmentation","wine"]
    input = [8,6,12,9,19,11]
    hidden_layer = [30,10,14,15,20,18]
    outputs = [29,4,1,1,7,1]

    for i in range(len(sets)):
        data_aba = Data_Processing([sets[i]], [], {})
        data_aba.load_data("./processed")


        #slice in to 5
        data_aba.slicer(5, sets[i])

        test_data = data_aba.file_array[0]
        training_data = data_aba.combine(data_aba.file_array[1:])

        print(sets[i])
        mlp_dims = [input[i],hidden_layer[i],outputs[i]]
        de = Differential_Evolution(100,mlp_dims,1.5,training_data,test_data)
        #best = de.run(10)
        score, guesses = de.metrics()
        print(guesses)
        print(score)
