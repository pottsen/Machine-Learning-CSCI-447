from K_Nearest import nearest_k_points
import pandas as pd


def k_means(k, dataframe):
    #shuffled_sliced_training = dataframe[1].values.tolist()
    shuffled_sliced_training = dataframe.copy()
    print('----------')
    #print(dataframe.iloc(0))

    if(k>len(shuffled_sliced_training)):
        return shuffled_sliced_training

    k_clusters = shuffled_sliced_training[0:k]
    # place k centroids randomly
    centroids=[] #list of all centroids
    clusters={} # = [index:[list of points assigned to this centroid's index],]
    for index, row in k_clusters.iterrows():
        centroids.append(index)
        #print(index)
        clusters.update({index:[]})
        #print(i)


    for index, row in shuffled_sliced_training.iterrows(): # assign data points to nearest centroid
        this_centroid = nearest_k_points(1,k_clusters,row)
        cent_index = this_centroid[0][0]
        clusters[cent_index].append(row)
        #cent_index = centroids.index(this_centroid)
        #print(cent_index)
        #clusters[cent_index].append(row)

    old_clusters = k_clusters.copy()
    old_clusters['1'].iloc[1] = 4
    #print(old_clusters)
    #print(k_clusters)
    iterations = 0

    while(not old_clusters.equals(k_clusters) and iterations<10): #recompute  data point assignment until centroids no longer move
        old_clusters = k_clusters.copy()
        #k_clusters = []

        cent_id = 0
        for centroid_idx, list1 in clusters.items():
            # print('#cluster list#'+str(centroid_idx))
            # print(list1)
            df = pd.DataFrame(list1)
            #print(df)
            num_points = 0
            new_centroid = []
            #print(df)
            #pd.to_numeric(df)
            classlabel=dataframe['0'].value_counts().idxmax()
            new_centroid = df.mean(axis = 0)
            new_centroid['0']=classlabel
            #print(centroid_idx)
            #print(k_clusters.iloc[1])
            # print("#new centroid#")
            # print(new_centroid)
            k_clusters.iloc[cent_id] = new_centroid
            #for point in list1:# for each point in centroid cluster, recompute distance by mean of points
            #    if (num_points == 0):
            #        new_centroid=point
            #    else:
            #        for i in range(len(new_centroid)):
            #            new_centroid[i]=new_centroid[i]+points[i]
            #    num_points+=1
            #for i in range(len(new_centroid)):
            #    new_centroid[i]=new_centroid[i]/num_points
            #k_clusters.append(new_centroid)
            cent_id+=1
        print('#K clusters#')
        print(k_clusters)
        #print('-----')
        #print(old_clusters)
        iterations+=1

        for index, row in clusters.items():
            clusters[index]=[]

        for index, row in shuffled_sliced_training.iterrows(): # assign data points to nearest centroid
            this_centroid = nearest_k_points(1,k_clusters,row)
            cent_index = this_centroid[0][0]
            clusters[cent_index].append(row)


    dataframe = pd.DataFrame(k_clusters)

    return dataframe
