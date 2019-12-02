from data_processing import Data_Processing
from mlp import MLP

def main():
    data_aba = Data_Processing(["abalone"], [8], {"M":"1", "F":"2", "I":"3"})
    #either:
    data_aba.process_data("./data")
    data_aba.write_data("./processed")
    #or:
    data_aba.load_data("./processed")
    #loads data into Data_Processing


    #demo of MLP prediction:
    '''
    mlp = MLP([8,18,28])
    print(mlp)

    tp = data_aba.file_array['abalone'][0][1:]
    print(tp)
    prediction = mlp.predict([tp],False)
    print(prediction)
    '''

    #demo of population manager:
    '''
    population = PopulationManager(5,[4,5,3])
    for i in population.population:
        print("-----------------")
        print(i)
        print("-----------------")
    '''

if __name__ == "__main__":
    main()
