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
    while runFull == True:
        #for left over data points associate each to the closest medoid by using distance
        medoid_dictionary = {}
        for index, row in training_data.iterrows():
            #find the closest medoid
            closest_medoid = nearest_k_points(1, medoids, row)
            closest_medoid_index = closest_medoid[0][0]
            #store lists of data points assigned to that medoid in a dictionary
            try: #add traning data
                medoid_dataset = medoid_dictionary[closest_medoid_index] 
                medoid_dictionary[closest_medoid_index].update(medoid_dataset, index)

            except:
                medoid_dictionary.update({closest_medoid_index:index})

        #Swap to false so it wil not be rerun unnless a medoid is swapped below
        runFull = False

        for key in medoid_dictionary:
            #initialize minimum cost to large value
            minimum_cost = 1000000000000000

            #include medoid in the cluster
            cluster_points = medoids[key]

            #add points mapped to medoid to cluster
            for value in medoid_dictionary[key]:
                for index in training_data:
                    if value == index:
                        cluster_points = cluster_points.append(training_data[index])

            #calculate cost of the medoid
            # medoid_distances = nearest_k_points(len(cluster_points),cluster_points, medoids[key])
            # for point in medoid_distances:
            #         minumum_cost = minimum_cost + medoid_distances[test_point][1]

            # cluster_points = cluster_points.append(medoids[key])

            #calculate cost of all points in cluster and update medoid if cost is less than medoid cost
            #have counter k=0 since first point will be medoid
            k=0
            for test_point in cluster_points:
                #resets cost for each point in cluster
                cost = 0

                #returns array of the points with the distances to each from from the test point
                all_point_distance_array = nearest_k_points(len(cluster_points)-1,cluster_points, test_point)

                #add up costs to get total
                for point in all_point_distance_array:
                    cost = cost + all_point_distance_array[test_point][1]
                
                #if new cost is less than medoid cost or previous, update
                if cost < minimum_cost and k > 0:
                    minimum_cost = cost
                    #will need to rerun full medoid if a point is swapped
                    runFull = True

                    #swap out medoied with data point that has lower cost
                    temp_medoid = medoids[key]
                    medoids[key] = test_point
                    training_data[test_point] = temp_medoid

                k = k + 1

             



        
        # #check all points associated to that medoid to see if one has a lower cost than the current medoid

        # #if a point has lower cost than medoid,then update that to be the medoid

        # #if a point was swapped with medoid rerun initial total assignment and then recheck if any assigned points are better
        #     totalCost = totalCost + closest_medoid[0][1]
        #     medoidList[closest_medoid[0][0]].add
        #     #do we need to check if classes line up?



    







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