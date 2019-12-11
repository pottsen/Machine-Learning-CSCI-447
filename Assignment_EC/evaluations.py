"""
Contains the fscore and MSE funtions for evaluation
"""

def f_score(guesses): #list of tuples [(actual, guess),(actual,guess)]
    confusion = {} #confusion matrix

    unique_classes = []
    for i in guesses:
        if i[0] not in unique_classes:
            unique_classes.append(i[0])

    #for each class, initialize the confusion matrix with zeros for that class
    for class_name in unique_classes:
        confusion.update({class_name:{'TP':0,'FP':0,'TN':0,'FN':0}})#class_name is the key for each classes confusion matrix
        #confusion{class:{TP:0,FP:0,TN:0,FN:0}}

    #for each class
    for class_name in unique_classes:
        #for each data point guessed in that class
        for result in guesses: #result[0] is actual class and result[1] is our guess
            if class_name == result[1] and class_name == result[0]: #guess is accurate with what the class actually was
                value = 'TP'
            if class_name == result[1] and class_name != result[0]: #guessed that a record was part of a class and it wasn't
                value = 'FP'
            if class_name != result[1] and class_name == result[0]: #guessed that a record was not part of a class and it was
                value = 'FN'
            if class_name != result[1] and class_name != result[0]: #guess is accurate that the record did not belong to a class
                value = 'TN'
            confusion[class_name][value] += 1 #increment that classes TP/FP/TN/FN count accordingly

    #calculate our class independent accuracy
    correct = 0
    total = 0
    for result in guesses:
        if(result[0]==result[1]):
            correct+=1
        total+=1
    accuracy = correct/total


    num_of_classes = len(confusion)

    count = 0
    precision = 0
    recall=0
    f1=0
    for class1, matrix in confusion.items():
        TP = matrix['TP']
        TN = matrix['TN']
        FP = matrix['FP']
        FN = matrix['FN']
        if((TP+FP) != 0):
            precision += TP/(TP+FP)
            ptemp = TP/(TP+FP)
        else:
            ptemp = 0
        if((TP+FN) != 0):
            recall += TP/(TP+FN)
            rtemp = TP/(TP+FN)
        else:
            rtemp = 0
        if((ptemp+rtemp)!=0):
            f1 += 2*ptemp*rtemp/(ptemp+rtemp)
        count+=1
    precision = precision/count
    recall = recall/count
    f1 = f1/count

    #f1 = 2*precision*recall/(precision+recall)

    metrics = {'F1': f1, 'Precision':precision, 'Recall':recall, 'Accuracy': accuracy}
    return metrics

def mse(guesses):
    error = 0
    for i in guesses:
        error += (i[0] - i[1])**2
    error/=len(guesses)
    return error
