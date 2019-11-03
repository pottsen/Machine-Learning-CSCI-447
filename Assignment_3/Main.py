from Data_Processing_Lists import Data_Processing_Lists
from Data_Processing_Pd import Data_Processing_Pd
from MLP import MLP


def main():
    

    data_aba = Data_Processing_Pd("abalone", 0, "./data")
    data_aba.strings_to_specific_num({"M":"1", "F":"2", "I":"3"})
    data_aba.shuffle_rows_df()
    data_aba.write_df_csv("./processed", "auto")
    


    data_array = Data_Processing_Lists("./processed", "abalone_processed")
    data_array.file_array = data_array.file_array[:500]
    number_of_classes = 3


    #data_as_2dList, number_of_hidden_layers, number_of_hidden_nodes_in_each_layer
    mlp = MLP(data_array.file_array, number_of_classes, 2, [5, 5])


    #testing feed forward
    l1 = np.array([5,2,1])
    l2 = np.array([1,1,7,1])
    weights = np.array([1,1,7,1],[1,1,7,1],[1,1,7,1],[1,1,7,1])
    mlp.feed_forward_layer


    
    

if __name__ == "__main__":
    main()

