from population_manager import PopulationManager
from data_processing import Data_Processing
import random
import math
import copy
from evaluations import  f_score, mse

class Genetic_Algorithm(PopulationManager):
    def __init__(self, pop_size, mlp_dims, number_of_generations, test_data, training_data):
        PopulationManager.__init__(self, pop_size, mlp_dims)
        self.test_data = test_data
        self.training_data = training_data
        self.number_of_inputs = mlp_dims[1]
        self.number_of_outputs = mlp_dims[-1]
        self.gen_replacement_rate = .5
        self.number_of_generations = number_of_generations

        
    #calculates a fitness for each indiv. in pop.
    def calculate_fitness(self):
        for i in self.population:
            outputs = []
            inputs = []
            #classification
            if self.number_of_outputs > 1:
                for j in self.training_data:
                    actual_class = j[0]
                    temp = [0] * self.number_of_outputs
                    temp[int(actual_class-1)] = 1
                    inputs.append(j[1:])
                    outputs.append(temp)
            #regression
            else:
                for j in self.training_data:
                    actual_class = j[0]
                    inputs.append(j[1:])
                    outputs.append(actual_class)
            i.fitness(inputs, outputs)
    
    
    def find_weakest(self):
        weakest1 = 10000000
        weakest2 = 10000000
        for i in range(len(self.population)):
            if self.population[i].individual_fitness < weakest1:
                weakest1 = self.population[i].individual_fitness
                weakest1_index = i
        for i in range(len(self.population)):
            if self.population[i].individual_fitness < weakest2 and i != weakest1_index:
                weakest2 = self.population[i].individual_fitness
                weakest2_index = i
        return weakest1_index, weakest2_index

    def find_fittest(self):
        fittest = 0
        for i in range(len(self.population)):
            if self.population[i].individual_fitness > fittest:
                fittest = self.population[i].individual_fitness
                fittest_index = i
        return fittest_index


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

        number_of_generations = 10000

        for j in range(self.number_of_generations):
            #update fitnesses 
            self.calculate_fitness()

            #provide some visibilty
            if j%50 == 0:
                fitnesses = []
                for k in self.population:
                    fitnesses.append(k.individual_fitness)
                print("gen ", j, "Fitness = ", fitnesses)

            for i in range(math.ceil(number_to_replace/2)):
                #select
                individual1 = self.russian_wheel_selection()
                individual2 = self.russian_wheel_selection()
                while individual1 == individual2:
                    individual2 = self.russian_wheel_selection()

                #cross-over
                child1, child2 = self.uniform_cross(self.population[individual1], self.population[individual2])

                #mutation
                child_mut1 = self.mutation(child1)
                child_mut2 = self.mutation(child2)

                #replace weakest
                weak1, weak2 = self.find_weakest()
                self.population[weak1].rezip_neuron(child_mut1)
                self.population[weak2].rezip_neuron(child_mut2)

        
        for i in self.test_data:
            #if regression
            if self.number_of_outputs == 1:
                results = ("gen:", j, ", fittest: ", self.population[self.find_fittest()], "mse: ",  mse( self.population[self.find_fittest].predict(i)))
            
            elif self.number_of_outputs > 1:
                results = ("gen:", j, ", fittest: ", self.population[self.find_fittest()], "f-score: ",  f_score(self.population[self.find_fittest].predict(i)))
                
        results_file = open("./results/results_ga.txt", "a+")
        results_file.write(results)
        results_file.close()
      
def test_ga():
    #Prepping data ----------------------------
    data_aba = Data_Processing(["abalone",], [8], {"M":"1", "F":"2", "I":"3"})

    data_aba.load_data("./processed")
    
    #triming it down to 10
    data_aba.file_array['abalone'] = data_aba.file_array['abalone'][:2000]
    
    #slice in to 5
    data_aba.slicer(5, "abalone")

    test_data = data_aba.file_array[0]
    training_data = data_aba.combine(data_aba.file_array[1:])
    #end --------------------------------------

    #tesing alogorithm ------------------------
    number_of_outputs = 29
    number_of_inputs = (len(data_aba.file_array[0][0][1:]))
    ga = Genetic_Algorithm(5, [number_of_inputs,30,number_of_outputs], 1000, test_data, training_data)
    ga.run_genetic_algorithm()
    
    #end --------------------------------------





if __name__ == "__main__":
    #data = Data_Processing(["abalone","car","forestfires","machine","segmentation","wine"], [8,6,12,9,0,11], {})
    #data.load_data("./processed")

    data_aba = Data_Processing(["abalone",], [8], {})
    data_car = Data_Processing(["car",], [6], {})
    data_img = Data_Processing(["segmentation",], [0], {})
    data_mach = Data_Processing(["machine",], [9], {})
    data_ff = Data_Processing(["forestfires",], [12],{})
    data_wine = Data_Processing(["wine",], [11], {})

    #LOAD DATA
    data_aba.load_data("./processed")
    data_car.load_data("./processed")
    data_img.load_data("./processed")
    data_mach.load_data("./processed")
    data_ff.load_data("./processed")
    data_wine.load_data("./processed")



    number_generation = 1000
    population = 50

    # -------Abalone-------
    results_file = open("./results/results_ga.txt", "a+")
    results_file.write("-------Abalone Reults-------")
    print("-------Abalone Reults-------")
    results_file.close()
    data_aba.slicer(5, "abalone")
    number_of_inputs = (len(data_aba.file_array[0][0][1:]))
    number_of_outputs = 29
    dim = [number_of_inputs,30,number_of_outputs]
    test_data = data_aba.file_array[0]
    training_data = data_aba.combine(data_aba.file_array[1:])

    ga = Genetic_Algorithm(population, dim, number_generation, test_data, training_data)
    ga.run_genetic_algorithm()


    # -------Car-------
    results_file = open("./results/results_ga.txt", "a+")
    results_file.write("-------Car Reults-------")
    print("-------Car Reults-------")
    results_file.close()
    data_car.slicer(5, "car")
    number_of_inputs = (len(data_car.file_array[0][0][1:]))
    number_of_outputs = 4
    dim = [number_of_inputs,10,number_of_outputs]
    test_data = data_car.file_array[0]
    training_data = data_car.combine(data_car.file_array[1:])

    ga = Genetic_Algorithm(population, dim, number_generation, test_data, training_data)
    ga.run_genetic_algorithm()


    # -------Forest-------
    results_file = open("./results/results_ga.txt", "a+")
    results_file.write("-------Forest Reults-------")
    print("-------Forest Reults-------")
    results_file.close()
    data_ff.slicer(5, "forestfires")
    number_of_inputs = (len(data_ff.file_array[0][0][1:]))
    number_of_outputs = 1
    dim = [number_of_inputs,20,number_of_outputs]
    test_data = data_ff.file_array[0]
    training_data = data_ff.combine(data_ff.file_array[1:])

    ga = Genetic_Algorithm(population, dim, number_generation, test_data, training_data)
    ga.run_genetic_algorithm()



    # -------Machine-------
    results_file = open("./results/results_ga.txt", "a+")
    results_file.write("-------Machine Reults-------")
    print("-------Machine Reults-------")
    results_file.close()
    data_img.slicer(5, "machine")
    number_of_inputs = (len(data_img.file_array[0][0][1:]))
    number_of_outputs = 1
    dim = [number_of_inputs,15,number_of_outputs]
    test_data = data_img.file_array[0]
    training_data = data_img.combine(data_img.file_array[1:])

    ga = Genetic_Algorithm(population, dim, number_generation, test_data, training_data)
    ga.run_genetic_algorithm()




    # -------Img/Seg-------
    results_file = open("./results/results_ga.txt", "a+")
    results_file.write("-------Img/Seg Reults-------")
    print("-------Img/Seg Reults-------")
    results_file.close()
    data_img.slicer(5, "segmentation")
    number_of_inputs = (len(data_img.file_array[0][0][1:]))
    number_of_outputs = 1
    dim = [number_of_inputs, 18, number_of_outputs]
    test_data = data_img.file_array[0]
    training_data = data_img.combine(data_img.file_array[1:])

    ga = Genetic_Algorithm(population, dim, number_generation, test_data, training_data)
    ga.run_genetic_algorithm()




    # -------Wine-------
    results_file = open("./results/results_ga.txt", "a+")
    results_file.write("-------Wine Reults-------")
    print("-------Wine Reults-------")
    results_file.close()
    data_wine.slicer(5, "wine")
    number_of_inputs = (len(data_wine.file_array[0][0][1:]))
    number_of_outputs = 1
    dim = [number_of_inputs, 14, number_of_outputs]
    test_data = data_wine.file_array[0]
    training_data = data_wine.combine(data_wine.file_array[1:])

    ga = Genetic_Algorithm(population, dim, number_generation, test_data, training_data)
    ga.run_genetic_algorithm()


    













    #end --------------------------------------
