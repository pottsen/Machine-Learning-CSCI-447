import pandas as pd
from K_Nearest import nearest_k_points

def condensed_k_nearest(k, training_data):
    new_data_point = True # Keeps track of whether or not a new point was added to condensed
    condensed = [] # Will contain the condensed set of training_data
    #print("Length of test data: ", len(test_data))
    #print(len(training_data) > 0)
    while new_data_point == True:
        new_data_point = False
        #print("looping")
        for i in range(len(training_data)): # Takes each item in the test data, and finds the nearest neighbor in the training data
##            print(i)
            print(type(training_data))
            nearest = nearest_k_points(k, training_data, training_data.iloc[i]) # Finds k nearest neighbors 
            guesses = [] # stores the class of each of the points in nearest
            for j in range(len(nearest)):
                guesses.append(nearest[j][-1]) # append class of each point in nearest
            guess = max(set(guesses), key = guesses.count)
            if training_data.iloc[i][0] != guess and not df_row_in_list(training_data.iloc[i], condensed): #Is the test data class the same as the nearest neighbor's class? 
                condensed.append(training_data.iloc[i])
                new_data_point = True # New datapoint added to condensed, continue while loop
    return condensed

def df_row_in_list(df_row, my_list):
    for item in my_list:
        if df_row.equals(item):
            return True
    return False
