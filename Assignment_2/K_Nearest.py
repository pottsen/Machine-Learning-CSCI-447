#choose an odd K for two classes
#k cannot be a multiple of the number of classes

import pandas as pd

#input k -> how many nearest points to return
#input dataframe -> the dataset to search in
#input example -> 
#return -> [[index, 1st_closest_distence], [index, 2nd_closest_distence]...]
def nearest_k_points(k, dataframe, example):
    if (len(example)) != (len(dataframe.iloc[0])):
        raise Exception ("example and dataframe row are not the same length")
    if (len(dataframe)) < k :
        raise Exception ("k number is smaller than our training data set")
    
    #find distence from example to point
    distences = []
    for index, row in dataframe.iterrows():
        distence = 0
        #dont use the first column = class
        for i in range(len(row)):
            distence += (float(example[i]) - float(row[i])) ** 2
        distences.append([index, distence])
    
    #find k number of smallest distences from distences
    closest = []
    for i in range(k):
        smallest = distences[0][1]
        smallest_list = distences[0]
        for index_distence in range(len(distences)):
            if smallest > distences[index_distence][1]:
                smallest_list = distences[index_distence]
        
        closest.append(smallest_list)
        distences.remove([smallest_list[0], smallest_list[1]])
    
    return closest


#makes all slices in dataframe one unified dataframe
#input-> a sliced dataframe
def concat_df(sliced_dataframes):
    if (len(sliced_dataframes)) < 2:
        return sliced_dataframes
    single_dataframe = pd.concat([sliced_dataframes[0], sliced_dataframes[1]])
    for i in range(len(sliced_dataframes)-2):
        single_dataframe = pd.concat([single_dataframe, sliced_dataframes[i+2]])
    return single_dataframe

#should make it so we only pass in a split data frame and a k
def k_nearst_neighbor(k, dataframes):
    test_data_index = 0
    test_data = dataframes[0][1].pop(test_data_index)
    training_data = concat_df(dataframes[0][1])

    #take each row of our test_data and find KNN in training_data
    for i in range(len(test_data)):
        k_closest = nearest_k_points(k, training_data, test_data.iloc[i])
        print(k_closest)

        
        

    