from Data_Processing_Pd import Data_Processing_Pd
from K_NearestNeighbor import K_Nearest_Neigbor, k_Nearest_Points
from Data_Processing_Lists import Data_Processing_Lists
from K_Medoids import k_medoids
from Loss_Functions import Loss_Functions
from Edited_NN import edited_nn
from K_Means import k_means


def main():
    data_aba = Data_Processing_Pd("abalone", 0, "./data")
    data_aba.strings_to_specific_num({"M":"1", "F":"2", "I":"3"})
    data_aba.shuffle_rows_df()
    data_aba.write_df_csv("./processed", "auto")


    data_array = Data_Processing_Lists("./processed", "abalone_processed")
    
    #data_array.file_array = data_array.file_array[:500]

    #test Array
    #data_array.file_array = [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],[15]]
    
    data_array.slicer(10)
    test_data = data_array.file_array.pop(0)
    data_array.join_array()
    training_data = data_array.file_array

    #------KNN Test--------
    results = K_Nearest_Neigbor(13, training_data, test_data, "class")
    loss = Loss_Functions(results)
    accuracy = loss.accuracy()
    print("Normal KNN: ", accuracy)
    loss.confusion_matrix_generator()

    #-----Edited KNN Test--------
    training_data = edited_nn(13, training_data)
    results = K_Nearest_Neigbor(13, training_data, test_data, "class")
    loss = Loss_Functions(results)
    accuracy = loss.accuracy()
    print("Edited Accuracy ", accuracy)


    # data_array = Data_Processing_Lists("./processed", "abalone_processed")
    # data_array.file_array = data_array.file_array[:500]
    # data_array.slicer(5)
    # test_data = data_array.file_array.pop(0)
    # data_array.join_array()
    # data_array.slicer(4)
    # medoids = data_array.file_array.pop(0)
    # data_array.join_array()
    # training_data = data_array.file_array
    # k_medoids(medoids, training_data)



    

if __name__ == "__main__":
    main()
