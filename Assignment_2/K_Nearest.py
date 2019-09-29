#choose an odd K for two classes
#k cannot be a multiple of the number of classes


#input k -> how many nearest points to return
#input dataframe -> the dataset to search in
#return -> [[index, 1st_closest_distence], [index, 2nd_closest_distence]...]
def nearest_k_points(k, dataframe):
    distences = []
    closest = []
    for index, row in dataframe.iterrows():
        distence = 0
        for num in row:
            distence += num * num
        distences.append([index, distence])
    print(distences)

    for i in range(k):
        smallest = distences[0][1]
        for index_distence in range(len(distences)):
            if smallest > distences[index_distence][1]:
                smallest_list = distences[index_distence]
        
        closest.append(smallest_list)
        distences.remove([smallest_list[0], smallest_list[1]])

    return closest

