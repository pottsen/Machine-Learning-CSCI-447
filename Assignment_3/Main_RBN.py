from Data_Processing_Lists import Data_Processing_Lists
from Data_Processing_Pd import Data_Processing_Pd
from RBN import RBN
from Loss_Functions import Loss_Functions
import numpy as np
import copy
from Edited_NN import edited_nn
from Condensed_K_Nearest import condensed_k_nearest
from K_Means import k_means
from K_Medoids import k_medoids


# def main():
def run():
    #in order to process the data, run the data processing file
    df_list = ["abalone", "car", "segmentation", "machine", "forestfires", "wine"]
    # df_list = ["machine", "forestfires", "wine"]
    df_class_num = [3, 4, 7, 1, 1, 1]
    # df_class_num = [1, 1, 1]

    results_file = open("./results/results_rbn.txt", "a+")
    for i in range(len(df_list)):

        data_array = Data_Processing_Lists("./processed", df_list[i]+"_processed")
        data_array.file_array = data_array.file_array[:]
        class_list = []
        for j in range(df_class_num[i]):  #makes an array of integers the same lenght as the number of classes each data set has
            class_list.append(j)
        #data.append(data_array)
        #classes.append(class_list)

        data_array.slicer(5)
        test_data = data_array.file_array.pop(0)
        data_array.join_array()
        training_data = data_array.file_array
        toy = copy.deepcopy(data_array)
        toy.slicer(4)
        medoids = toy.file_array.pop(0)
        toy.join_array()
        training_data_toy = toy.file_array
        
        knn = [edited_nn(13,training_data), k_means(100, training_data), k_medoids(medoids, training_data_toy)]
        algo_name = ['edited knn', 'kmeans', "kmedoids"]
        algo_idx = 0
        for centers in knn:
            for k in range(1):
                
                # def __init__(self, data, output, gaussian_function_type, centers):
                print("class list", class_list)
                print("size rbf\n", len(centers))
                print(df_list[i])

                rbn = RBN(training_data, class_list, 1, centers)
                rbn.train()

                guesses = rbn.classify(test_data)
                losses = Loss_Functions(guesses)

                if (len(class_list)==1):#regression
                    print_str = "MSE for "+str(df_list[i])+" fold: "+str(k)+ " \n"+algo_name[algo_idx]
                    print(print_str)
                    print(losses.mse())

                    results_file.write("\n"+print_str)
                    results_file.write("\nMSE: "+str(losses.mse())+"\n")
                
                else:#classification
                    losses.confusion_matrix_generator()
                    print_str = "Fscore for "+str(df_list[i])+" fold: "+str(k)+ "\n"+algo_name[algo_idx]
                    print(print_str)
                    print(losses.f_score())

                    results_file.write("\n"+print_str)
                    results_file.write("\nF-score: "+str(losses.f_score())+"\n")
            algo_idx+=1
    results_file.close()

    # if __name__ == "__main__":
    #     main()

run()
