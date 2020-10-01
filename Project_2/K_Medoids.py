import pandas as pd
from K_Nearest import nearest_k_points, concat_df, k_nearest_neighbor_regression
#from Main import shuffle_pd_df, slicer, slice_pd_df_using_np, load_data_from_csv


def k_medoids(medoids, training_data):    
    
    #convert data to pandas data frames
    training_data_df = pd.DataFrame(training_data)
    medoids_df = pd.DataFrame(medoids)

    #initialize count so medoids loop doesnt run forever
    count = 0
    #flag to say if we should run through medoid algorith
    runFull = True
    while runFull == True and count < 50:
        #for left over data points associate each to the closest medoid by using distance
        #make dictionary to assign associated points to medoids
        medoid_dictionary = {}
        #iterate through training data
        for index, row in training_data_df.iterrows():
            #find the closest medoid
            closest_medoid = nearest_k_points(1, medoids_df, row)
            #store index of the closest medoid
            closest_medoid_index = closest_medoid[0][0]
            #store lists of data points assigned to that medoid in a dictionary
            #add traning data
            try: 
                medoid_dictionary[closest_medoid_index].append(index)

            except:
                medoid_dictionary.update({closest_medoid_index:[index]})

        #Swap to false so it wil not be rerun unnless a medoid is swapped below
        runFull = False
        count = count + 1

        for key in medoid_dictionary:
            #initialize minimum cost
            minimum_cost = 0

            #include medoid in the cluster
            cluster_points = [medoids_df.loc[key]]

            #add points mapped to medoid to cluster
            for training_index in medoid_dictionary[key]:
            
                cluster_points.append(training_data_df.loc[training_index])

            #convert data to pandas data frames
            cluster_points_df = pd.DataFrame(cluster_points)

            #counter used because we want to initialize cost to medoid cost
            k=0
            for index, test_point in cluster_points_df.iterrows():
                #resets cost for each point in cluster
                cost = 0

                #returns array of the points with the distances to each from from the test point
                all_point_distance_array = nearest_k_points(len(cluster_points),cluster_points_df, test_point)

                #add up costs to get total
                for point in range(len(all_point_distance_array)):
                    cost = cost + all_point_distance_array[point][1]

                #set cost to medoid cost first
                if k==0:
                    minimum_cost = cost
                    k = k + 1
                    
                #if new cost is less than medoid cost or previous, update
                if cost < minimum_cost:
                    minimum_cost = cost
                    
                    #will need to rerun full medoid if a point is swapped
                    runFull = True

                    #swap out medoied with data point that has lower cost
                    temp_medoid = medoids_df.loc[key]
                    medoids_df.loc[key] = training_data_df.loc[index]
                    training_data_df.loc[index] = temp_medoid

    return medoids_df