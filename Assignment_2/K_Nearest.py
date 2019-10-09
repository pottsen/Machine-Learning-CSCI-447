#choose an odd K for two classes
#k cannot be a multiple of the number of classes

import pandas as pd

#input k -> how many nearest points to return
#input dataframe -> the dataset to search in
#input example -> 
#return -> [[index, 1st_closest_distance, class], [index, 2nd_closest_distance, class]...]
def nearest_k_points(k, dataframe, example):
    if (len(example)) != (len(dataframe.iloc[0])):
        raise Exception ("example and dataframe row are not the same length")
    if (len(dataframe)) < k :
        raise Exception ("k number is smaller than our training data set")
    
    #find distance from example to point
    distances = []
    for index, row in dataframe.iterrows():
        distance = 0
        row_class = row[0]
        #dont use the first column = class
        for i in range(len(row)-1):
            distance += (float(example[i+1]) - float(row[i+1])) ** 2
        distances.append([index, distance, row_class])
    
    
    #find k number of smallest distances from distances
    closest = []
    for i in range(k):
        smallest = distances[0][1]
        smallest_list = distances[0]
        for index_distance in range(len(distances)):
            if smallest > distances[index_distance][1]:
                #return the class
                smallest = distances[index_distance][1]
                smallest_list = distances[index_distance]
        
        closest.append(smallest_list)
        distances.remove([smallest_list[0], smallest_list[1], smallest_list[2]])
    
    return closest


#makes all slices in dataframe one unified dataframe
#input-> a sliced dataframe
def concat_df(sliced_dataframes):
    if (len(sliced_dataframes)) < 2:
        return sliced_dataframes
    single_dataframe = pd.concat([sliced_dataframes[0], sliced_dataframes[1]], sort=False)
    for i in range(len(sliced_dataframes)-2):
        single_dataframe = pd.concat([single_dataframe, sliced_dataframes[i+2]], sort=False)
    return single_dataframe

#should make it so we only pass in a split data frame and a k
#def k_nearest_neighbor(k, dataframes):
def k_nearest_neighbor(k, training_data, test_data):
  
    all_guesses = []
    #take each row of our test_data and find KNN in training_data
    #is that collumns or rows??
    for i in range(len(test_data)):
        k_closest = nearest_k_points(k, training_data, test_data.iloc[i])
        
        actual_class = test_data.iloc[i, 0]
        #average the class
        guesses = []
        for j in range(k):
            guesses.append(k_closest[j][-1])
        
        guesses = max(set(guesses), key = guesses.count) 

        #print('actual = '+actual_class+' predicted = '+guesses)

        all_guesses.append([actual_class, guesses])
    
    return all_guesses
