from Data_Processing_Lists import Data_Processing_Lists
from Data_Processing_Pd import Data_Processing_Pd
from MLP import MLP
import numpy as np
import copy


def main():


    data_aba = Data_Processing_Pd("abalone", 0, "./data")
    data_aba.strings_to_specific_num({"M":"1", "F":"2", "I":"3"})
    data_aba.shuffle_rows_df()
    data_aba.write_df_csv("./processed", "auto")

    data_car = Data_Processing_Pd("car", 0, "./data")
    data_car.strings_to_specific_num({"vhigh":"4", "high":"3", "med":"2", "low":"1", "5more":"5", "more":"6" 
    ,"small":"1", "big":"3"})
    data_car.shuffle_rows_df()
    data_car.write_df_csv("./processed", "auto")
    
    data_img = Data_Processing_Pd("segmentation", 0, "./data")    
    data_img.shuffle_rows_df()
    data_img.write_df_csv("./processed", "auto")

    data_mach = Data_Processing_Pd("machine", 0, "./data")
    data_mach.shuffle_rows_df()
    data_mach.write_df_csv("./processed", "auto")

    data_ff = Data_Processing_Pd("forestfires", 0, "./data")
    data_ff.strings_to_specific_num({"jan":"1", "feb":"2", "mar":"3", "apr":"4", "may":"5", "jun":"6", "jul":"7",
    "aug":"8", "sep":"9", "oct":"10", "nov":"11", "dec":"12", "sun":"1", "mon":"2",
    "tue":"3", "wed":"4", "thu":"5", "fri":"6", "sat":"7"})
    data_ff.shuffle_rows_df()
    data_ff.write_df_csv("./processed", "auto")    

    data_wine = Data_Processing_Pd("wine", 0, "./data")
    data_wine.shuffle_rows_df()
    data_wine.write_df_csv("./processed", "auto")

    df_list = ["abalone", "car", "segmentation", "machine", "forestfires", "wine"]
    df_class_num = [3, 4, 7, 1, 1, 1]

    #data = []
    #classes = []

    results_file = open("./results/results.txt", "a+")
    for i in range(len(df_list)):

        data_array = Data_Processing_Lists("./processed", df_list[i]+"_processed")
        data_array.file_array = data_array.file_array[:]
        class_list = []
        for j in range(len(df_class_num)):  #makes an array of integers the same lenght as the number of classes each data set has
            class_list.append(j)
        #data.append(data_array)
        #classes.append(class_list)

        data_array.slicer(5)
        
        layer_num = 2
        layer_nodes = [10,12]

        for k in range(5):
            
            toy = copy.deepcopy(data_array)

            test_data = toy.file_array.pop(k)
            toy.join_array()
            training_data = toy.file_array
    
            mlp = MLP(training_data, class_list, layer_num, layer_nodes, True)
            mlp.train()
            #mlp = MLP(training_data, class_list, 1, [12], True)
            guesses = mlp.classify_batch(test_data)
            losses = Loss_Functions(guesses)

            if (len(class_list)==1):#regression
                print("MSE for",df_list[i],"fold:",k, "\n Network layer dimensions",layer_nodes)
                print(losses.mse())

                results_file.write("\nMSE for",df_list[i],"fold:",k, "\n Network layer dimensions",layer_nodes)
                results_file.write(losses.mse())
            
            else:#classification
                losses.confusion_matrix_generator()
                print("Fscore for",df_list[i],"fold:",k, "\n Network layer dimensions",layer_nodes)
                print(losses.fscore())

                results_file.write("\nFscore for",df_list[i],"fold:",k, "\n Network layer dimensions",layer_nodes)
                results_file.write(losses.fscore())

    results_file.close()
            
    """
    #XOR = np.array([[0, 1, 1],[1, 1, 0],[1, 0, 1],[0, 0, 0]])
    #mlp = MLP(XOR, [0,1], 1, [5])
    #mlp.train()

    #data_as_2dList, possible_outputs, number_of_hidden_layers, number_of_hidden_nodes_in_each_layer
    mlp = MLP(training_data, [1,2,3], 2, [10,12], True)
    # mlp = MLP(data_array.file_array[:49], [1,2,3], 1, [12], True)
    # mlp = MLP(data_array.file_array[:49], [1,2,3], 0, [], True)
    # mlp = MLP(data_array2.file_array[:49], [1], 2, [10,12], False)
    # mlp = MLP(data_array2.file_array[:49], [1], 1, [12], True)
    # mlp = MLP(data_array2.file_array[:49], [1], 0, [], False)
    mlp.train()

    test = data_array.file_array[-1:][0]
    # test2 = data_array2.file_array[-1:][0]
    test_class = test[0]
    # test_class2 = test2[0]
    test=test[1:]
    # test2=test2[1:]
    print(test_class)
    mlp.classify(test)
    """

if __name__ == "__main__":
    main()
