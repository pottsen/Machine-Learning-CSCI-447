from K_NearestNeighbor import k_Nearest_Points, classification_guess

def edited_nn(k, training_data):
    #go through every row
    i = 0
    while(len(training_data) != i):
        #take a row out of the training data
        example = training_data.pop(i)
        #find the rows closest points
        closest_points = k_Nearest_Points(k, training_data, example)
        #find what KNN classified it as
        guess = classification_guess(closest_points)
        actual = example[0]
        #if the row was classified correctly, add it back in the list
        if (guess == actual):
            training_data.insert(i, example)
            i += 1
    return training_data


            


            



