import pandas as pd
from K_Nearest import nearest_k_points

def condensed_k_nearest(k, training_data, test_data):
    condensed = [] # Condensed set of points to be returned
    new_data_point = True # Keeps track of whether or not a new point was added to condensed
    while new_data_point == True:
        new_data_point = False
        for i in range(len(test_data)): # Takes each item in the test data, and finds the nearest neighbor in the training data
            nearest = nearest_k_points(k, training_data, test_data.iloc[i]) # Finds k nearest neighbors
            for j in nearest: # Go through each of the k nearest neighbors.
                if test_data.iloc[i][0] != j[2]: #Is the test data class the same as the nearest neighbor's class?
                    condensed.append(training_data.loc[j[0]]) # Add the item to condensed
                    training_data.drop(labels = j[0]) # Remove the item from training data
                    new_data_point = True # New datapoint added to condensed, continue loop                                          
    return condensed
