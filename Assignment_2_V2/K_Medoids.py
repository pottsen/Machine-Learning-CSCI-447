import numpy as np
import pandas as pd
from K_NearestNeighbor import k_Nearest_Points, K_Nearest_Neigbor
#from Main import shuffle_pd_df, slicer, slice_pd_df_using_np, load_data_from_csv


def k_medoids(medoids, training_data):    
    
    #convert data to pandas data frames
    # training_data_np = pd.DataFrame(training_data).to_numpy()
    # medoids_np = pd.DataFrame(medoids).to_numpy()

    #initialize count so medoids loop doesnt run forever
    count = 0
    #flag to say if we should run through medoid algorith
    runFull = True
    while runFull == True and count < 50:
        #for left over data points associate each to the closest medoid by using distance
        #make dictionary to assign associated points to medoids
        medoid_dictionary = {}
        print("count ", count)
        #iterate through training data
        for row in range(len(training_data)):
            #find the closest medoid
            closest_medoid = k_Nearest_Points(1, medoids, training_data[row])
            print("Clostest Medoid ", closest_medoid)
            #store index of the closest medoid
            index = closest_medoid[0][2]
            #store lists of data points assigned to that medoid in a dictionary
            #add traning data
            try: 
                medoid_dictionary[index].append(row)

            except:
                medoid_dictionary.update({index:[row]})

        #Swap to false so it wil not be rerun unnless a medoid is swapped below

        runFull = False
        count = count + 1
        """
        medoids_to_remove = []
        training_data_to_medoid = []
        for key in medoid_dictionary:
            #initialize minimum cost
            minimum_cost = 0

            #include medoid in the cluster
            cluster_points = [medoids[key]]

            #add points mapped to medoid to cluster
            indices = []
            for training_index in medoid_dictionary[key]:
                indices.append(training_index)
                cluster_points.append(training_data[training_index])
            
            
            #counter used because we want to initialize cost to medoid cost
            k=0
            for index in range(len(cluster_points)):
                #resets cost for each point in cluster
                cost = 0

                #returns array of the points with the distances to each from from the test point
                all_point_distance_array = k_Nearest_Points(len(cluster_points),cluster_points, cluster_points[index])
                print("Distance point array: ", all_point_distance_array[0])
                print(len(all_point_distance_array))

                #add up costs to get total
                for point in range(len(all_point_distance_array)):
                    cost = cost + all_point_distance_array[point][1]
                print("Cost ", cost)

                #set cost to medoid cost first
                if k==0:
                    minimum_cost = cost
                    k = k + 1
                    print("Medoid Cost ", minimum_cost)
                    
                #if new cost is less than medoid cost or previous, update
                if cost < minimum_cost:
                    minimum_cost = cost
                    print("New lowest cost ", minimum_cost)
                    #will need to rerun full medoid if a point is swapped
                    runFull = True

                    #swap out medoied with data point that has lower cost
                    try: 
                        medoids_to_remove.append(key)
                        training_data_to_medoid.append(indices[index])

                    except:
                        medoids_to_remove = [key]
                        training_data_to_medoid = [indices[index]]
        for i in training_data_to_medoid:
            medoids.append(training_data[i])
        for i in medoids_to_remove:
            training_data.append(medoids[i])
        # medoids.append(training_data[training_data_to_medoid)
        # training_data.append(medoids[medoids_to_remove])
        medoids_to_remove.sort()
        training_data_to_medoid.sort()
        for i in reversed(medoids_to_remove):
            medoids.pop(i)
        for i in reversed(training_data_to_medoid):
            training_data.pop(i)


    return medoids

def distance_calc(k, training, data_point):
    if (len(data_point)) != len(training[0]):
        raise Exception ("Example and dataframe row are not the same length")
    if len(training) < k:
        raise Exception("k number of closest points is larger than our training data set")

    distances = np.zeros([len(training),1])
    print(distances)

    for i in range(len(training)):
        for j in range(len(data_point)):
            distances[i] += (data_point[j] - training[i,j])**2
    sorted_distances = np.sort(distances, axis=0)
    print(sorted_distances[0])        
    print("Training ", training[0])
    training_with_distances = np.append(training, distances, axis = -1)
    print("Training with distance ", training_with_distances)
    # training_with_distances = np.sort(training_with_distances[:,:], axis = -1)
    # training_with_distances = np.argsort(training_with_distances[:], axis=-1)
    training_with_distances = training_with_distances[training_with_distances[:,-1].argsort()]
    print("Sorted training with distance: ", training_with_distances)
    training_with_distances = training_with_distances[:k,:]
    print(training_with_distances)
    return training_with_distances
    """