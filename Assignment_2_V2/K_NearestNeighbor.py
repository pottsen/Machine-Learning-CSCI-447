from Data_Processing_Pd import Data_Processing_Pd

#input-> traing data as 2d list, example as list
#return-> closest_points = [[label, distance, index]...]
def k_Nearest_Points(k, training, example):
    #make sure the length of the example is the length of a row our data set
    if (len(example)) != (len(training[0])):
        raise Exception ("example and dataframe row are not the same length", (len(example)), len(training[0]))
    if (len(training)) < k :
        raise Exception ("k number is smaller than our training data set")
    distance = []
    for i in range(len(training)):
        single_distance = 0
        for j in range(len(example)):
            #[label, distance, index]
            single_distance += (example[j] - training[i][j]) ** 2
        distance.append([training[i][0], single_distance, i])
    closest_points = []
    
    
    for i in range(k):
        closest = distance[0]
        for j in range(len(distance)):
            if closest[1] > distance[j][1]:
                closest = distance[j]
        closest_points.append(closest)
        distance.remove(closest)
    print("Closest Points Ordered: ", closest_points)
    return closest_points

#input-> training data as 2d list, test data as 2d list
#return-> [[actual, guess, index], ...]
def K_Nearest_Neigbor(k, training_data, test_data):
    results = []
    for i in range(len(test_data)):
        closest_points = k_Nearest_Points(k, training_data, test_data[i])
        guesses = []
        for j in range(len(closest_points)):
            guesses.append(closest_points[j][0])
        guess = max(set(guesses), key = guesses.count) 
        actual = test_data[i][0]
        index = test_data[i][2]
        results.append([actual, guess, index])
    print(results)
    return results


