from Data_Processing_Lists import Data_Processing_Lists
from Data_Processing_Pd import Data_Processing_Pd
from MLP import MLP
import numpy as np


def main():


    data_aba = Data_Processing_Pd("abalone", 0, "./data")
    data_aba.strings_to_specific_num({"M":"1", "F":"2", "I":"3"})
    data_aba.shuffle_rows_df()
    data_aba.write_df_csv("./processed", "auto")



    data_array = Data_Processing_Lists("./processed", "abalone_processed")
    data_array.file_array = data_array.file_array[:4000]
    number_of_classes = 3

    #XOR = np.array([[0, 1, 1],[1, 1, 0],[1, 0, 1],[0, 0, 0]])
    #mlp = MLP(XOR, [0,1], 1, [5])
    #mlp.train()

    #data_as_2dList, possible_outputs, number_of_hidden_layers, number_of_hidden_nodes_in_each_layer
    mlp = MLP(data_array.file_array[:3999], [1,2,3], 2, [10,12], True)
    # mlp = MLP(data_array.file_array[:49], [1,2,3], 1, [12], True)
    # mlp = MLP(data_array.file_array[:49], [1,2,3], 0, [], True)
    mlp.train()
    test = data_array.file_array[-1:][0]
    #test = np.array([1, 0, 1])
    test_class = test[0]
    test=test[1:]
    print(test_class)
    mlp.classify(test)


if __name__ == "__main__":
    main()
