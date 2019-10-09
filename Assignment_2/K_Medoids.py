import pandas as pd
from K_Nearest import nearest_k_points, concat_df
#from Main import shuffle_pd_df, slicer, slice_pd_df_using_np, load_data_from_csv


def k_medoids(medoids, training_data):    
    # #pull k random data points from training data to be medoids
    # #randomize the training data
    # shuffled_training = shuffle_pd_df(dataframes)
    # #slice into sections for medoids and training set
    # shuffled_sliced_training = slicer(4, shuffled_training)
    # #set medoids
    # medoids = shuffled_sliced_training.pop(0)
    # #set training data
    # training_data = concat_df(shuffled_sliced_training)
    
    training_data_df = pd.DataFrame(training_data)
    medoids_df = pd.DataFrame(medoids)
    #print(training_data)
    #print(training_data_df)
    #print(len(medoids))
    #print(medoids_df)
    
    count = 0
    runFull = True
    while runFull == True and count < 10:
        #for left over data points associate each to the closest medoid by using distance
        #make dictionary to assign associated points to medoids
        medoid_dictionary = {}
        for index, row in training_data_df.iterrows():
            #find the closest medoid
            closest_medoid = nearest_k_points(1, medoids_df, row)
            #print("closest medoid ", closest_medoid)
            closest_medoid_index = closest_medoid[0][0]
            #store lists of data points assigned to that medoid in a dictionary
            try: #add traning data
                medoid_dictionary[closest_medoid_index].append(index)

            except:
                medoid_dictionary.update({closest_medoid_index:[index]})

        print(medoid_dictionary)
        #Swap to false so it wil not be rerun unnless a medoid is swapped below
        runFull = False
        count = count + 1

        for key in medoid_dictionary:
            print("key ", key)
            #initialize minimum cost to large value
            minimum_cost = 0

            #include medoid in the cluster
            cluster_points = [medoids_df.iloc[key]]

            #add points mapped to medoid to cluster
            for value in medoid_dictionary[key]:
                print("training data", value)
                cluster_points.append(training_data_df.iloc[value])
                # for index, row in training_data_df.iterrows():
                #     if value == index:
                #         cluster_points.append(training_data_df.iloc[index])

            cluster_points_df = pd.DataFrame(cluster_points)
            #calculate cost of all points in cluster and update medoid if cost is less than medoid cost
            #have counter k=0 since first point will be medoid
            print(cluster_points_df)
    return medoids
"""
            k=0
            for index, test_point in cluster_points_df.iterrows():
                #resets cost for each point in cluster
                cost = 0

                #returns array of the points with the distances to each from from the test point
                print("# of cluster points", len(cluster_points))
                all_point_distance_array = nearest_k_points(len(cluster_points),cluster_points_df, test_point)
                print(all_point_distance_array)

                #add up costs to get total
                for point in range(len(all_point_distance_array)):
                    cost = cost + all_point_distance_array[point][1]
                    print("cost", cost)
                #if new cost is less than medoid cost or previous, update
                if k==0:
                    minimum_cost = cost
                    k = k + 1
                    
                print("minimum cost", minimum_cost)
                if cost < minimum_cost:
                    minimum_cost = cost
                    #will need to rerun full medoid if a point is swapped
                    runFull = True

                    #swap out medoied with data point that has lower cost
                    temp_medoid = medoids[key]
                    medoids[key] = test_point
                    print("new medoid", medoids[key])
                    training_data_df.iloc[test_point.index] = temp_medoid
                    print("new training", training_data[test_point.index])

"""
             


# def test():

#     #processes all data and store in procecessed folder
#     #DONT RUN EVERY TIME
#     #process_data()

#     #load processed data into dataframes 
#     data_frames = load_data_from_csv("./processed/machine_processed")

#     #cut the data into ten for validation
#     #data_frames = [[(String)name, [[slice1][slice2][slice3][sliceN]]], ...]
#     # number_of_sections = 1
#     # data_frames = slice_pd_df_using_np(number_of_sections, data_frames)
    
#     returned_medoids = k_medoids(1, data_frames)

#     print(returned_medoids)

#     # define our K Values
#     # k = [13, 37,61]
#     # folds = number_of_sections
#     # for num in k:    
#     #     #for each file

#     #     #perform the nearest neighbor algorithm
#     #     cross_validation(folds, num, data_frames[0],'k-nn')

#     #     #Test EditedK_Neatest
#     #     cross_validation(folds, num, data_frames[4] , "edited")


    

# test()
# return 