from population_manager import PopulationManager
from data_processing import Data_Processing


class Genetic_Algorithm(PopulationManager):
    def __init__(self, pop_size, mlp_dims, test_data, training_data):
        PopulationManager.__init__(self, pop_size, mlp_dims)
        self.test_data = test_data
        self.training_data = training_data








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
    ga = Genetic_Algorithm(5, [(len(data_aba.file_array[0][0][1:])),30,29], test_data, training_data)
    print(len(data_aba.file_array[0][0][1:]))
    #ga.population[0].print_weights()

    actual_classes = []
    inputs = []
    for i in training_data:
        inputs.append(ga.population[0].predict(i[1:]))
        actual_classes.append(i[0])
    print(inputs)
    print(len(inputs[0]))

    fitness = ga.population[0].fitness(inputs, actual_classes)
    print(fitness)
    
    #end --------------------------------------
