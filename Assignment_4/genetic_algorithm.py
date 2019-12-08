from population_manager import PopulationManager
from data_processing import Data_Processing
import random


class Genetic_Algorithm(PopulationManager):
    def __init__(self, pop_size, mlp_dims, test_data, training_data):
        PopulationManager.__init__(self, pop_size, mlp_dims)
        self.test_data = test_data
        self.training_data = training_data
        self.number_of_inputs = mlp_dims[1]
        self.number_of_outputs = mlp_dims[-1]

        
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
    
    

    def russian_wheel_selection(self):
        #add up all the fitness
        #randomly select a number btw 0-fitness sum
        #select which ever individual it lands on
        
        #sum up the fitness
        fitness_sum = 0
        for i in self.population:
            fitness_sum += i.individual_fitness
        
        #pick a radom float between 0 and fitness sum
        selection = random.uniform(0, fitness_sum)
        #when the fitness sum is bigger or equal to the random number, select individ
        fitness_sum = 0
        for i in self.population:
            fitness_sum += i.individual_fitness
            if selection <= fitness_sum:
                return i
        


    def cross_over(self):
        pass

    def run_genetic_algorithm(self):

        self.calculate_fitness()
        individual1 = self.russian_wheel_selection()
        individual2 = self.russian_wheel_selection()
        while individual1.layers[0] == individual2.layers[0]:
            individual2 = self.russian_wheel_selection()
        
        child = self.uniform_cross(individual1, individual2)
        self.mutation(child)
        
        
        
    



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
