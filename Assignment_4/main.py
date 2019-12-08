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
    '''
    data_aba = Data_Processing(["abalone",], [8], {"M":"1", "F":"2", "I":"3"})
    #either:
    data_aba.process_data("./data")
    data_aba.write_data("./processed")
    #or:
    data_aba.load_data("./processed")
    '''
    #loads data into Data_Processing

    a = {"M":"0", "F":"1", "I":"2"}
    c = {"unacc":"0","acc":"1","vgood":"3","good":"2",  "5more":"5","more":"6",  "big":"2","small":"0", "vhigh":"3", "high":"2", "med":"1", "low":"0"}
    #f = {"jan":"0.866,0.5","feb":"0.5,0.866","mar":"0,1","apr":"-0.5,0.866","may":"-0.866,0.5","jun":"-1,0","jul":"-0.866,-0.5","aug":"-0.5,-0.866","sep":"0,-1","oct":"0.5,-0.866","nov":"0.866,0.5","dec":"1,0"}
    #f2 = {"sun":"0.623,0.782","mon":"-0.222,0.975","tue":"-0.901,0.434","wed":"-0.901,-0.434","thu":"-0.223,-0.975","fri":"0.623,-0.782","sat":"1,0"}
    f = {"jan":"0","feb":"1","mar":"2","apr":"3","may":"4","jun":"5","jul":"6","aug":"7","sep":"8","oct":"9","nov":"10","dec":"11","sun":"0","mon":"1","tue":"2","wed":"3","thu":"4","fri":"5","sat":"6"}
    s = {'BRICKFACE':"0", 'SKY':"1", 'FOLIAGE':"2", 'CEMENT':"3", 'WINDOW':"4", 'PATH':"5", 'GRASS':"6"}
    m = {}

    a.update(c)
    a.update(f)
    a.update(s)
    #input sizes = []
    #output sizes = []
    #data_aba = Data_Processing(["abalone","car","forestfires","machine","segmentation","wine"], [8,6,12,9,0,11], {})
    #machine is the one where we want an auto generated dictionary
    data_aba = Data_Processing(["abalone","car","forestfires","segmentation","wine"], [8,6,12,0,11], a)

    #either:
    data_aba.process_data("./data")
    data_aba.write_data("./processed")



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
