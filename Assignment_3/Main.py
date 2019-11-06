from Data_Processing_Lists import Data_Processing_Lists
from Data_Processing_Pd import Data_Processing_Pd
from MLP import MLP
import numpy as np


def main():


    data_aba = Data_Processing_Pd("abalone", 0, "./data")
    data_aba.strings_to_specific_num({"M":"1", "F":"2", "I":"3"})
    data_aba.shuffle_rows_df()
    data_aba.write_df_csv("./processed", "auto")

    data_ff = Data_Processing_Pd("forestfires", 0, "./data")
    data_ff.strings_to_specific_num({"jan":"1", "feb":"2", "mar":"3", "apr":"4", "may":"5", "jun":"6", "jul":"7",
    "aug":"8", "sep":"9", "oct":"10", "nov":"11", "dec":"12", "sun":"1", "mon":"2", 
    "tue":"3", "wed":"4", "thu":"5", "fri":"6", "sat":"7"})
    data_ff.shuffle_rows_df()
    data_ff.write_df_csv("./processed", "auto")



    data_array = Data_Processing_Lists("./processed", "abalone_processed")
    data_array.file_array = data_array.file_array[:4000]
    number_of_classes = 3

    data_array2 = Data_Processing_Lists("./processed", "forestfires_processed")
    data_array2.file_array = data_array2.file_array[:40]
    number_of_classes = 1

    #XOR = np.array([[0, 1, 1],[1, 1, 0],[1, 0, 1],[0, 0, 0]])
    #mlp = MLP(XOR, [0,1], 1, [5])
    #mlp.train()

    #data_as_2dList, possible_outputs, number_of_hidden_layers, number_of_hidden_nodes_in_each_layer
    # mlp = MLP(data_array.file_array[:3999], [1,2,3], 2, [10,12], True)
    # mlp = MLP(data_array.file_array[:49], [1,2,3], 1, [12], True)
    # mlp = MLP(data_array.file_array[:49], [1,2,3], 0, [], True)
    mlp = MLP(data_array2.file_array[:49], [1], 2, [10,12], False)
    # mlp = MLP(data_array2.file_array[:49], [1], 1, [12], True)
    # mlp = MLP(data_array2.file_array[:49], [1], 0, [], False)
    mlp.train()

    test = data_array.file_array[-1:][0]
    test2 = data_array2.file_array[-1:][0]
    #test = np.array([1, 0, 1])
    test_class = test[0]
    test_class2 = test2[0]
    test=test[1:]
    test2=test2[1:]
    print(test_class2)
    mlp.classify(test2)


if __name__ == "__main__":
    main()
