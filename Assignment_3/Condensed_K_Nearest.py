import pandas as pd
import random
from K_NearestNeighbor import k_Nearest_Points, classification_guess

def condensed_k_nearest(k, training_data):
    random.shuffle(training_data)
##    print("randomized training set")
    new_data_point = True # Keeps track of whether or not a new point was added to condensed
    condensed = [] # Will contain the condensed set of training_data

    while new_data_point == True: # Stops looping once nothing else gets added to condensed.
        new_data_point = False

        for i in range(len(training_data)): # Takes each item in the training data, and finds the nearest neighbor in the training data
            nearest = k_Nearest_Points(k, training_data, training_data[i]) # Finds k nearest neighbors 
            guess = classification_guess(nearest)
            if training_data[i][0] != guess and not df_row_in_list(training_data[i], condensed): #Is the data point actual class different from the guessed class? and is the point not already contained in condensed?
                condensed.append(training_data[i]) # 
                new_data_point = True # New datapoint added to condensed, continue while loop
    return condensed

def df_row_in_list(df_row, my_list): # Returns true if df_row exists in my_list, and false otherwise.
    for item in my_list:
        if df_row == item:
            return True
    return False
