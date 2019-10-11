from K_Nearest import nearest_k_points, concat_df, k_nearest_neighbor_regression
import pandas as pd


def k_means(k, dataframe):
    shuffled_sliced_training = dataframe.copy()
    print('----------')

    #if our k value is greater than the training data, we already have as many centroids as we need
    if(k>len(shuffled_sliced_training)): 
        return shuffled_sliced_training

    k_clusters = shuffled_sliced_training[0:k]
    # place k centroids randomly (the first k points which are randomized)

    centroids=[] #list of all centroids
    clusters={} # = [index:[list of points assigned to this centroid's index]...]
    for index, row in k_clusters.iterrows():
        centroids.append(index)
        clusters.update({index:[]})

    # assign training data points to the nearest centroid to them
    for index, row in shuffled_sliced_training.iterrows(): 
        this_centroid = nearest_k_points(1,k_clusters,row)
        cent_index = this_centroid[0][0]
        clusters[cent_index].append(row)

    #we want to run this until it converges so we need to test k_clusters against another dataframe
    old_clusters = k_clusters.copy()
    #change a value in old_clusters so that it is not equal to k_clusters on the first run through the while loop
    old_clusters['1'].iloc[1] = 4

    #this is used to get the indexes of the centroids
    iterations = 0

    #recompute data point assignment until centroids no longer move
    while(not old_clusters.equals(k_clusters) and iterations<20):  #if centroids are not re-adjusted, then old_clusters = k_clusters
        old_clusters = k_clusters.copy()

        cent_id = 0
        #for each cluster
        for centroid_idx, list1 in clusters.items():

            #make a new dataframe of all of the points classified to the current cluster (used for finding centroid)
            df = pd.DataFrame(list1)
            
            num_points = 0
            new_centroid = []
            
            #sometimes the clusters dont have points classified to them, so here we avoid an error from that
            try:

                #find the most popular class label in this cluster
                #not using this anymore as the cluster would be dominated by the common class
                classlabel=df['0'].value_counts().idxmax()

                #find the center of the points in our cluster and set that to new_centroid
                new_centroid = df.mean(axis = 0)

                #re-set the center's label to 0 so we can set it later
                new_centroid['0'] = 0

                #find the 5 nearest points to the center of the cluster
                points = (nearest_k_points(5,df,new_centroid))
                
                #set the class label of the centroid to the most popular of the 5 nearest neighbors
                label = []
                for i in points:
                    label.append(i[2])
                label = max(set(label), key = label.count)
                new_centroid['0']=label
                
                #add the new centroid to k_clusters
                k_clusters.iloc[cent_id] = new_centroid
                #print(new_centroid)

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

        #re-assign every training point to one of our new centroids
        for index, row in shuffled_sliced_training.iterrows():
            this_centroid = nearest_k_points(1,k_clusters,row)
            cent_index = this_centroid[0][0]
            clusters[cent_index].append(row)


    #set the dataframe to the clusters we generated and return that set
    dataframe = pd.DataFrame(k_clusters)

    return dataframe
