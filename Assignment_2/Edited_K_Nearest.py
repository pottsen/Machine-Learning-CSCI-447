from K_Nearest import nearest_k_points


def edited_k_nearest(k, training_data):
    #go through the training data

    #we are using k+1 becuase, the closest point is obviously itself
    
    for i in range(100):
        pre_triaining_data = training_data

        for index, row in training_data.iterrows():
            print(k)
            k_closest = nearest_k_points(k, training_data, row)
            k_closest = k_closest[1:]
            print(k_closest)
            print("our traing")
            guesses = []
            for j in range(len(k_closest)):
                guesses.append(k_closest[j][-1])
            
            guess = max(set(guesses), key = guesses.count) 

            #determine actual class
            actual_class = row[0]

            if actual_class != guess:
                training_data.drop(index)
                
                #print(index , " dropped")
            # else:
            #     print(index , " kept")
            if (len(training_data) <= k):
                return training_data
        if (len(training_data)) == (len(pre_triaining_data)):
            return training_data
    return training_data   
