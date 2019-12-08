from population_manager import PopulationManager
from data_processing import Data_Processing
import random
import math


class Genetic_Algorithm(PopulationManager):
    def __init__(self, pop_size, mlp_dims, test_data, training_data):
        PopulationManager.__init__(self, pop_size, mlp_dims)
        self.test_data = test_data
        self.training_data = training_data
        self.number_of_inputs = mlp_dims[1]
        self.number_of_outputs = mlp_dims[-1]
        self.gen_replacement_rate = .5

        
    #calculates a fitness for each indiv. in pop.
    def calculate_fitness(self):
        for i in self.population:
            outputs = []
            inputs = []
            for j in self.training_data:
                actual_class = j[0]
                temp = [0] * self.number_of_outputs
                temp[int(actual_class-1)] = 1
                inputs.append(j[1:])
                outputs.append(temp)
            i.fitness(inputs, outputs)
    
    
    #picks an indiv from the pop. with highly fit indiv more likley to be chosen
    def russian_wheel_selection(self):
        
        #sum up the fitness
        fitness_sum = 0
        for i in self.population:
            fitness_sum += i.individual_fitness
        
        #pick a radom float between 0 and fitness sum
        selection = random.uniform(0, fitness_sum)

        #when the fitness sum is bigger or equal to the random number, select individ
        fitness_sum = 0
        for i in range(len(self.population)):
            fitness_sum += self.population[i].individual_fitness
            if selection <= fitness_sum:
                return i
        

    def run_genetic_algorithm(self):


        number_to_replace = int(len(self.population) * self.gen_replacement_rate)

        number_of_generations = 100

        for j in range(number_of_generations):

            self.calculate_fitness()
            count = 0
            for k in self.population:
                print(count, "---", k.individual_fitness)
                count+=1

            for i in range(math.ceil(number_to_replace/2)):
                #select
                individual1 = self.russian_wheel_selection()
                individual2 = self.russian_wheel_selection()
                while individual1 == individual2:
                    individual2 = self.russian_wheel_selection()

                #cross-over
                child1, child2 = self.uniform_cross(self.population[individual1], self.population[individual2])

                #mutation
                child1.rezip_neuron(self.mutation(child1))
                child2.rezip_neuron(self.mutation(child2))

                #replace selected old gen. with new gen.
                self.population[individual1] = child1
                self.population[individual2] = child2

            


if __name__ == "__main__":

    #Prepping data ----------------------------
    data_aba = Data_Processing(["abalone",], [8], {"M":"1", "F":"2", "I":"3"})

    data_aba.load_data("./processed")
    
    #triming it down to 10
    data_aba.file_array['abalone'] = data_aba.file_array['abalone'][:10]
    
    #slice in to 5
    data_aba.slicer(5, "abalone")

    test_data = data_aba.file_array[0]
    training_data = data_aba.combine(data_aba.file_array[1:])
    #end --------------------------------------

    #tesing alogorithm ------------------------
    number_of_outputs = 29
    number_of_inputs = (len(data_aba.file_array[0][0][1:]))
    ga = Genetic_Algorithm(5, [number_of_inputs,30,number_of_outputs], test_data, training_data)
    ga.run_genetic_algorithm()
    
    #end --------------------------------------
