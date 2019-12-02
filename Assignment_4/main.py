from data_processing import Data_Processing
from population_manager import PopulationManager
from mlp import MLP

def main():
    data_aba = Data_Processing(["abalone"], [8], {"M":"1", "F":"2", "I":"3"})
    #either:
    data_aba.process_data("./data")
    data_aba.write_data("./processed")
    #or:
    data_aba.load_data("./processed")
    #loads data into Data_Processing


    #1: demo of MLP prediction:
    '''
    mlp = MLP([8,18,28])
    print(mlp)

    tp = data_aba.file_array['abalone'][0][1:]
    print(tp)
    prediction = mlp.predict([tp],False)
    print(prediction)'''
    #end 1------------------

    #2: demo of changing neuron weights (used to cross two neurons together easily)
    '''
    unz = mlp.unzip_neuron()
    print(unz)

    mlp.print_weights()
    unz[0] = .555
    unz[-1] = .555
    mlp.rezip_neuron(unz)
    mlp.print_weights()'''
    #end 2-------------------------------------

    #3: demo of population manager:
    '''
    population = PopulationManager(5,[4,5,3])
    for i in population.population:
        print("-----------------")
        print(i)
        print("-----------------")
    '''
    #end 3-------------------------------------

    #4: demo of population manager crossing :

    population = PopulationManager(2,[4,5,3])
    iter = 0
    for i in population.population:
        print('----organism ' + str(iter) +'----')
        print(i.print_weights())
        iter+=1
    child = population.uniform_cross(population.population[0],population.population[1])
    print('----child----')
    print(child.print_weights())

    #end 4-------------------------------------

if __name__ == "__main__":
    main()
