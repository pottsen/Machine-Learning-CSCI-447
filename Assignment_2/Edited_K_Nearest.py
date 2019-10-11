from K_Nearest import nearest_k_points


def edited_k_nearest(k, training_data):
    #if we cannot converge after 100 runs, return what we have
    for i in range(100):
        #go through the training data
        pre_triaining_data = training_data
        for index, row in training_data.iterrows():
            #find K closest points
            #we are using k+1 becuase, the closest point is obviously itself
            k_closest = nearest_k_points(k+1, training_data, row)
            #dont consider the first KNN results becuase the distence between a point and itself is 0
            k_closest = k_closest[1:]
            guesses = []
            #make an array of guesses
            for j in range(len(k_closest)):
                guesses.append(k_closest[j][-1])
            #find the most prevelent guess
            guess = max(set(guesses), key = guesses.count) 

            #determine actual class
            actual_class = row[0]

            #if the point was classified incorrectly, drop is from our training dataframe
            if actual_class != guess:
                training_data.drop(index)
            
            #if we ever get to a point where k is greater than the number of points in a training dataset, KNN cannot do its job properly
            if (len(training_data) <= k):
                return training_data
        #Once we have gone through the dataset and there are no more dropped point, return the edited training dataframe
        if (len(training_data)) == (len(pre_triaining_data)):
            return training_data
    return training_data   
