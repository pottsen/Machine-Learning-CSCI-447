from Data_Processing_Pd import Data_Processing_Pd

#input-> traing data as 2d list, example as list
#return-> closest_points = [[label, distence]...]
def k_Nearest_Points(k, training, example):
    #make sure the length of the example is the length of a row our data set
    if (len(example)) != (len(training[0])):
        raise Exception ("example and dataframe row are not the same length", (len(example)), len(training[0]))
    if (len(training)) < k :
        raise Exception ("k number is smaller than our training data set")
    distence = []
    for i in range(len(training)):
        single_distence = 0
        for j in range(len(example)):
            #[label, distence]
            single_distence += (example[j] - training[i][j]) ** 2
        distence.append([training[i][0], (example[j] - training[i][j]) ** 2])
    closest_points = []
    for i in range(k):
        closest = distence[0]
        for j in range(len(distence)):
            if closest[1] > distence[i][1]:
                closest = distence[i]
        closest_points.append(closest)
    return closest_points

#input training data as 2d list, test data as 2d list
def K_Nearest_Neigbor(k, training_data, test_data):
    results = []
    for i in range(len(test_data)):
        closest_points = k_Nearest_Points(k, training_data, test_data[i])
        guesses = []
        for j in range(len(closest_points)):
            guesses.append(closest_points[j][0])
        guess = max(set(guesses), key = guesses.count) 
        actual = test_data[i][0]
        results.append([actual, guess])
    print(results)
    return results


