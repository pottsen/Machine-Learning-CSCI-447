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
    while runFull == True and count < 100:
        #for left over data points associate each to the closest medoid by using distance
        #make dictionary to assign associated points to medoids
        medoid_dictionary = {}
        print("count ", count)
        #iterate through training data
        for row in range(len(training_data)):
            #find the closest medoid
            closest_medoid = k_Nearest_Points(1, medoids, training_data[row])
            # print("Clostest Medoid ", closest_medoid)
            #store index of the closest medoid
            index = closest_medoid[0][2]
            #store lists of data points assigned to that medoid in a dictionary
            #add traning data
            try: 
                medoid_dictionary[index].append(row)

            except:
                medoid_dictionary.update({index:[row]})

        #Swap to false so it wil not be rerun unnless a medoid is swapped below

        # print("Medoid Dictionary:")
        # print(medoid_dictionary)
        runFull = False
        count = count + 1

        medoids_to_remove = []
        training_data_to_medoid = []
        for key in medoid_dictionary:
            #initialize minimum cost
            minimum_cost = 0
            #include medoid in the cluster
            cluster_points = [medoids[key]]

            #add points mapped to medoid to cluster
            indices = [key]
            for training_index in medoid_dictionary[key]:
                indices.append(training_index)
                cluster_points.append(training_data[training_index])
            
            
            #counter used because we want to initialize cost to medoid cost
            k=0
            minimum_index = 0
            print("NEW CLUSTER")
            for index in range(len(cluster_points)):
                #resets cost for each point in cluster
                cost = 0
                all_point_distance_array = k_Nearest_Points(len(cluster_points),cluster_points, cluster_points[index])

                #add up costs to get total
                for point in range(len(all_point_distance_array)):
                    cost = cost + all_point_distance_array[point][1]
                # print("Cost ", cost)

                #set cost to medoid cost first
                if k==0:
                    minimum_cost = cost
                    k = k + 1
                    # print("Medoid Cost ", minimum_cost)
                    
                #if new cost is less than medoid cost or previous, update

                if cost < minimum_cost:
                    minimum_cost = cost
                    minimum_index = index
                    print("NEW LOWEST COST ---- index = ", indices[index])
                    # print("New lowest cost ", minimum_cost)
                    #will need to rerun full medoid if a point is swapped
                    

                    #swap out medoied with data point that has lower cost
            if minimum_index != 0:
                runFull = True
                print("ADDING KEY AND INDEX TO LIST")

                medoids_to_remove.append(key)
                print("MEDOID KEY LIST", medoids_to_remove)
                training_data_to_medoid.append(indices[minimum_index])
                print("TRIANING DATA KEY LIST", training_data_to_medoid)
                # try: 
                #     medoids_to_remove.append(key)
                #     print("MEDOID KEY LIST", medoids_to_remove)
                #     training_data_to_medoid.append(indices[minimum_index])
                #     print("TRIANING DATA KEY LIST", training_data_to_medoid)

                # except:
                #     medoids_to_remove = [key]
                #     # print("Index ", index)
                #     # print("# of cluster points", len(cluster_points))
                #     # print("Length of indices ", len(indices))
                #     training_data_to_medoid = [indices[index]]
            # print(training_data_to_medoid)
        print("Medoids: ", len(medoids))
        for i in range(len(medoids)):
            print(medoids[i])
        for i in training_data_to_medoid:
            # print("appended training data: ", training_data[i])
            new_medoid = training_data[int(i)]
            medoids.append(new_medoid)
        print("Appended Medoids: ", len(medoids))
        for i in range(len(medoids)):
            print(medoids[i])
        for i in medoids_to_remove:
            training_data.append(medoids[i])
        # medoids.append(training_data[training_data_to_medoid)
        # training_data.append(medoids[medoids_to_remove])
        medoids_to_remove.sort()
        print("MEDOIDS TO REMOVE ", medoids_to_remove)
        training_data_to_medoid.sort()
        print("TRAINING DATA TO MEDOIDS ", training_data_to_medoid)
        for i in reversed(medoids_to_remove):
            # print("medoid to remove: ", medoids[i])
            medoid_removed = medoids.pop(i)
            # print("medoid removed: ", medoid_removed)
        for i in reversed(training_data_to_medoid):
            training_data.pop(i)
        # print("Medoids")
        # for i in range(len(medoids)):
        #     print(medoids[i])
        # print("Length of medoids ", len(medoids))

    return medoids








# def distance_calc(k, training, data_point):
#     if (len(data_point)) != len(training[0]):
#         raise Exception ("Example and dataframe row are not the same length")
#     if len(training) < k:
#         raise Exception("k number of closest points is larger than our training data set")

#     distances = np.zeros([len(training),1])
#     print(distances)

#     for i in range(len(training)):
#         for j in range(len(data_point)):
#             distances[i] += (data_point[j] - training[i,j])**2
#     sorted_distances = np.sort(distances, axis=0)
#     print(sorted_distances[0])        
#     print("Training ", training[0])
#     training_with_distances = np.append(training, distances, axis = -1)
#     print("Training with distance ", training_with_distances)
#     # training_with_distances = np.sort(training_with_distances[:,:], axis = -1)
#     # training_with_distances = np.argsort(training_with_distances[:], axis=-1)
#     training_with_distances = training_with_distances[training_with_distances[:,-1].argsort()]
#     print("Sorted training with distance: ", training_with_distances)
#     training_with_distances = training_with_distances[:k,:]
#     print(training_with_distances)
#     return training_with_distances
