from K_Nearest import nearest_k_points
from Main import shuffle_pd_df, slicer, slice_pd_df_using_np

def k_medoids(k, dataframes):    
    #pull k random data points from training data to be medoids
    
    #randomize the training data
    shuffled_training = shuffle_pd_df(dataframes)
    #slice into sections for medoids and training set
    shuffled_sliced_training = slicer(4, shuffled_training)
    #set medoids
    medoids = shuffled_sliced_training[0]
    #set training data
    training_data = shuffled_sliced_training[1:]

    runFull = True
    while runFull = True:
        #for left over data points associate each to the closest medoid by using distance
        totalCost = 0.0
        medoidList[]
        for index, row in training_data.iterrows():
            #find the closest medoid
            closest_medoid = nearest_k_points(1, medoids, row)
            #append medoid index to training datum
            training_data.append[closest_medoid[0][0]]
            #store lists of data points assigned to that medoid
            
            #dictionary of medoids and associated points
        
        #check all points associated to that medoid to see if one has a lower cost than the current medoid

        #if a point has lower cost than medoid,then update that to be the medoid

        #if a point was swapped with medoid rerun initial total assignment and then recheck if any assigned points are better
            totalCost = totalCost + closest_medoid[0][1]
            medoidList[closest_medoid[0][0]].add
            #do we need to check if classes line up?



    







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