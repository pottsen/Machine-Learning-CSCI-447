import numpy as np
import pandas as pd
from K_NearestNeighbor import k_Nearest_Points, K_Nearest_Neigbor


def k_means(k, training_data):

    print('----------')

    #if our k value is greater than the training data, we already have as many centroids as we need
    if(k>len(training_data)):
        return training_data

    k_clusters = training_data[0:k]
    # place k centroids randomly (the first k points which are randomized)

    #centroids=[] #list of all centroids
    clusters={} # = [index:[list of points assigned to this centroid's index]...]
    i=0
    for point in k_clusters:
        #centroids.append(point)
        clusters.update({i:[]})
        i+=1

    # assign training data points to the nearest centroid to them
    for point in training_data:
        this_centroid = k_Nearest_Points(1,k_clusters,point)
        #print(this_centroid)
        cent_index = this_centroid[0][2]
        clusters[cent_index].append(point)
    #print(clusters)

    #we want to run this until it converges so we need to test k_clusters against another dataframe
    old_clusters = 0

    iterations = 0

    #recompute data point assignment until centroids no longer move
    while(not old_clusters==k_clusters and iterations<100):  #if centroids are not re-adjusted, then old_clusters = k_clusters
        old_clusters = k_clusters.copy()

        cent_id = 0
        #for each cluster
        for centroid_idx, list1 in clusters.items():

            #sometimes the clusters dont have points classified to them, so here we avoid an error from that
            try:


                new_centroid = list1[0].copy()
                num_points = 1
                for list_element in range(len(list1)-1): #for each point classified to the centroid
                    for item in range(len(list1[0])-1): #for each column (except for class)
                        new_centroid[item+1]+=list1[list_element+1][item+1]
                    num_points+=1
                for item in range(len(new_centroid)-1):
                    new_centroid[item+1] /= num_points
                #find the 5 nearest points to the center of the cluster
                points = (k_Nearest_Points(3,list1,new_centroid))
                #set the class label of the centroid to the most popular of the 5 nearest neighbors
                label = []
                for i in points:
                    label.append(i[0])
                label = max(set(label), key = label.count)
                new_centroid[0]=label
                k_clusters[centroid_idx] = new_centroid.copy()


            except:
                #print('no points assigned to this cluster')
                pass


            #used for indexing centroid points in k_clusters
            cent_id+=1


        #print('#K clusters#')
        #print(k_clusters)
        #print('-----')
        #print(old_clusters)

        #index iterations so we dont run this algorithm forever
        iterations+=1

        #clear the dictionary that keeps track of which points are closest to which centroids
        for index, row in clusters.items():
            clusters[index]=[]

        for point in training_data:
            this_centroid = k_Nearest_Points(1,k_clusters,point)
            #print(this_centroid)
            cent_index = this_centroid[0][2]
            clusters[cent_index].append(point)



    #set the dataframe to the clusters we generated and return that set

    training_data = k_clusters

    print(iterations)



    return training_data
