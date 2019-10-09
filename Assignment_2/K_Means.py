from K_Nearest import nearest_k_points
import pandas


def k_means(k, dataframe):
    shuffled_sliced_training = dataframe.values.tolist()
    #shuffled_sliced_training = dataframe

    if(k>len(shuffled_sliced_training)):
        return shuffled_sliced_training
    
    k_clusters = shuffled_sliced_training[0:k]
    # place k centroids randomly

    cluster_dict={} # = {centroid:[points associated with cetroid]}
    for i in k_clusters:
        cluster_dict.update({i:[]})

    for i in shuffled_sliced_training: # assign data points to nearest centroid
        this_centroid = nearest_k_points(1,k_clusters,row)
        cluster_dict[this_centroid].append(i)

    old_clusters = []
    iterations = 0

    while(old_clusters != k_clusters or iterations<10): #recompute  data point assignment until centroids no longer move
        old_clusters = k_clusters
        k_clusters = []

        for key, list1 in cluster_dict:
            num_points = 0
            new_centroid = []
            for point in list1:# for each point in centroid cluster, recompute distance by mean of points
                if (num_points == 0):
                    new_centroid=point
                else:
                    for i in range(len(new_centroid)):
                        new_centroid[i]=new_centroid[i]+points[i]
                num_points+=1
            for i in range(len(new_centroid)):
                new_centroid[i]=new_centroid[i]/num_points
            k_clusters.append(new_centroid)

        

    k_clusters = pd.DataFrame(k_clusters)

    return k_clusters
