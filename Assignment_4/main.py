from data_processing import Data_Processing
from population_manager import PopulationManager
from mlp import MLP

def f_score( guesses ): #list of tuples [(actual, guess),(actual,guess)]
    confusion = {} #confusion matrix

    unique_classes = []
    for i in guesses:
        if i[0] not in unique_classes:
            unique_classes.append(i)

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

def main():
    data_aba = Data_Processing(["abalone",], [8], {"M":"1", "F":"2", "I":"3"})
    #either:
    data_aba.process_data("./data")
    data_aba.write_data("./processed")
    #or:
    data_aba.load_data("./processed")
    #loads data into Data_Processing


    #1: demo of MLP prediction:
    '''
    mlp = MLP([8,18,28])
    print(mlp)

    tp = data_aba.file_array['abalone'][0][1:]
    print(tp)
    print(mlp.layers[0].get_layer_size())
    prediction = mlp.predict([tp])
    print(prediction)'''


    #end 1------------------

    #2: demo of changing neuron weights (used to cross two neurons together easily)
    '''
    unz = mlp.unzip_neuron()
    print(unz)

    mlp.print_weights()
    unz[0] = .555
    unz[-1] = .555
    mlp.rezip_neuron(unz)
    mlp.print_weights()'''
    #end 2-------------------------------------

    #3: demo of population manager:
    '''
    population = PopulationManager(5,[4,5,3])
    for i in population.population:
        print("-----------------")
        print(i)
        print("-----------------")
    '''
    #end 3-------------------------------------

    #4: demo of population manager crossing :
    '''
    population = PopulationManager(2,[4,5,3])
    iter = 0
    for i in population.population:
        print('----organism ' + str(iter) +'----')
        print(i.print_weights())
        iter+=1
    child = population.uniform_cross(population.population[0],population.population[1])
    print('----child----')
    print(child.print_weights())
    '''
    #end 4-------------------------------------

    #5: demo of fitness:
    '''
    mlp = MLP([4,5,3])
    print(mlp)

    prediction = mlp.predict([1,1,0,1])
    print(prediction)
    print(mlp.fitness([[1,1,0,1]],[[0,1,0]]))
    print(mlp.fitness([[1,1,0,1]],[[0,0,1]]))
    '''
    #end 5--------------------------------------

if __name__ == "__main__":
    main()
