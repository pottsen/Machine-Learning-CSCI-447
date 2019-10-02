from K_Nearest import nearest_k_points


def edited_k_nearest(k, training_data):
    #go through the training data

    #we are using k+1 becuase, the closest point is obviously itself
    for index, row in training_data.iterrows():
        k_closest = nearest_k_points(k+1, training_data, row)
        k_closest = k_closest[1:]
        guesses = []

        for j in range(k):
            guesses.append(k_closest[j][-1])
        
        guess = max(set(guesses), key = guesses.count) 

        #determine actual class
        actual_class = row[0]

        if actual_class != guess:
            training_data.drop(index)
            
    return training_data   


# return 