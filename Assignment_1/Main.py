#Machine Learning: Naive Bayes on 5 data sets
#Peter Ottsen, Bruce Clark, Forest Edwards, Justin McGowen


import os

#split into 10 even sections, counts of each class, frequency of attribute
#returns 10 lists of even lenght of records that are each are proportional to the origional data set's classes
def stratify(data,class_col): # file data, # of class column in data
    ten_strata = [[],[],[],[],[],[],[],[],[],[]]

    class_data = {} #dictionary of class_name:data of all records for that class
    for i in data:
        columns = i.replace('\n','').split(",")
        class_name = columns[class_col]
        try:
            data = class_data[class_name]
        except:
            data = ""
        class_data.update([(class_name,data+i)]) #add the data in i to our class (class is represented by the key)

    class_frequency = {}  # dictionary of class_name: number of times that class occurs in data
    for key in class_data: #find frequency of each class
        counts = class_data[key].count('\n')
        class_frequency.update([(key,counts)])

    for key in class_data: #assign records to each strata
        #fix rounding
        strata_subsize = class_frequency[key]/10
        for i in range(10):
            start_range = int(i * strata_subsize)
            end_range = int(i * strata_subsize + strata_subsize)
            records = class_data[key].split('\n')
            ten_strata[i]+=(records[start_range:end_range])
    
    #print(class_frequency)
    #print (ten_strata)

    return ten_strata, class_frequency


#3D dictionary
#attribute_prob = {key1:{key1':{key1":probability}},key2:{key2':{key2":probability}},...}
#where keyX = the class
# and keyX' = the attribute column
# and keyX" = the attribute value
#and probability = the probability of this feature(attribute) given the class
#F(keyX"|keyX) = attribute_prob[keyX][keyX'][keyX"]
def classAttributeFrequency(data_set, class_col, class_counts):
    attribute_counts = {} #see above
    for i in data_set: # i will represent each record of our data_set
        columns = i.split(",")
        class_name = columns[class_col]


        for j in range(len(columns)): #j will represent the index of each attribute
            if(j!=class_col): # should not count class occurances given class (==1)
                attribute_col = j
                attribute_value = columns[j]
                try: #if we have seen this attribute value before, add 1 to the count
                    att_count = attribute_counts[class_name][attribute_col][attribute_value]
                    attribute_counts[class_name][attribute_col][attribute_value] = att_count + 1
                    #attribute_counts[class_name][attribute_value]+=1 #simplification of last two lines
                except KeyError:
                    try:#if we have not seen this attribute value before, make the count 1
                        attribute_counts[class_name][attribute_col].update({attribute_value:1})
                    except KeyError:
                        try:#if we have not seen this attribute column (ie attribute) before initialize the attirbute
                            attribute_counts[class_name].update({attribute_col:{attribute_value:1}})
                        except KeyError: #if we have not seen this class before initialize the class
                            attribute_counts.update({class_name:{attribute_col:{attribute_value:1}}})                            
    attribute_probability = attribute_counts #initialize probability as counts because we will overwrite count with probability
    for key_class in attribute_probability:
        for key_attribute in attribute_probability[key_class]:
            for key_value in attribute_probability[key_class][key_attribute]:
                #attribute_count = str(attribute_probability[key_class][key_attribute][key_value])
                #conditional_probability = attribute_count +'/'+ str(class_counts[key_class]) #prints string of probability in fraction form
                attribute_count = attribute_probability[key_class][key_attribute][key_value] + 1  #laplace smoother here
                conditional_probability = attribute_count / (class_counts[key_class] +len(columns)-1)#prob = occurances of attribute given a class / total occurences of given class
                attribute_probability[key_class][key_attribute][key_value] = conditional_probability

    return attribute_probability

#returns the probability that an example is of a class based on our conditional probabilities 'attribute_probability'
def classify(example, attribute_probability, class_col, class_counts): 
    prob_calc = {} # Stores the probability of the example appearing in the dataset - probability:class
    columns = example.split(",") # Turns the string into a list, separated by commas
    #class_name = columns[class_col]
    for key_class in attribute_probability:
        #for j in range(len(columns)):
            #print("key class", key_class)
            #prob_calc[key_class] = class_frequency
        class_prob = class_counts[key_class]/class_counts['total'] #class frequency
        for example_col in range(len(columns)):
            if(example_col!=class_col):
                attr_value = columns[example_col]
                try:
                    conditional_prob = attribute_probability[key_class][example_col][attr_value]
                except KeyError: #do not want our probability to be zero if there was not instance of an attribute in our training set
                    conditional_prob = 1/(entry_num+len(columns)-1) #additive smoother
                class_prob = class_prob * conditional_prob
        prob_calc.update({class_prob:key_class})
    highest_prob = 0
    for key in prob_calc:
        if key>=highest_prob:
            highest_prob=key
    #print(prob_calc)
    return prob_calc[highest_prob]


#this will take a data set and return the number of occurances of each class in that data set as well as total number of entries in a dictionary
def classFrequency(data,class_col):
    class_freq = {}
    total = 0

    for i in data:
        columns = i.replace('\n','').split(",")
        class_name = columns[class_col]
        try:
            count = class_freq[class_name]
        except:
            count = 0
        class_freq.update([(class_name,count+1)]) #add the data in i to our class (class is represented by the key)
        total += 1
    class_freq.update([('total',total)])
    return class_freq

#training takes an array of 10 subdivided portions of our data set and tests on that set using 10-fold validation
def training(ten_strata, class_column):
    #print(ten_strata)
    training = []
    #ten fold validation
    loss = {}
    for j in range(10):# j is the index of the test data set in each round
        for i in range(len(ten_strata)):
            if(i!=j):
                training+=(ten_strata[i])
        #print(training)
        global entry_num 
        entry_num = len(training)
        test_class_counts = classFrequency(training, class_column)

        classAttributeFrequencies = classAttributeFrequency(training, class_column, test_class_counts)
        #print(classAttributeFrequencies)

        resultsFile = open('Results.txt','a+')

        confusion = {}#confusion matrix
        for class_name in classAttributeFrequencies:
            confusion.update({class_name:{'TP':0,'FP':0,'TN':0,'FN':0}})#class_name is the key for each classes confusion matrix
        #confusion{class:{TP:0,FP:0,TN:0,FN:0}}
        for i in ten_strata[j]:
            actual_class = i.split(',')[class_column]
            guess = classify(i, classAttributeFrequencies, class_column, test_class_counts)
            #print(guess)            
            for class_name in classAttributeFrequencies:
                if class_name == guess and class_name == actual_class: #guess is accurate with what the class actually was
                    value = 'TP'
                if class_name == guess and class_name != actual_class: #guessed that a record was part of a class and it wasn't
                    value = 'FP'
                if class_name != guess and class_name == actual_class: #guessed that a record was not part of a class and it was
                    value = 'FN'
                if class_name != guess and class_name != actual_class: #guess is accurate that the record did not belong to a class
                    value = 'TN'
                confusion[class_name][value] += 1 #increment that classes TP/FP/TN/FN count accordingly
        
        #print(confusion)
        

        for key in confusion:#writes each confusion matrix to file
            output = str(key)+" = "+str(confusion[key])
            resultsFile.write(output)
            resultsFile.write("\n")


        
        for class_name in classAttributeFrequencies:
            total = confusion[class_name]['TP'] + confusion[class_name]['FP'] + confusion[class_name]['TN'] + confusion[class_name]['FN']
            
            #calculates accuracy for given fold and class
            accuracy = (confusion[class_name]['TP'] + confusion[class_name]['TN'])/ total
            #print('Accuracy-'+class_name+": "+str(accuracy))
            resultsFile.write(' Accuracy-'+class_name+": "+str(accuracy))
            positives = confusion[class_name]['TP'] + confusion[class_name]['FP']
            
            #calculates precision for given fold and given class
            try:
                precision = confusion[class_name]['TP'] / positives
            except:
                precision = 'INF'
            #print('Precision-'+class_name+": "+str(precision))
            resultsFile.write(' Precision-'+class_name+": "+str(precision)+"\n")
            
            #adds all of our accuracies and precisions for eachfold together
            try:                
                loss[class_name]['Accuracy'] += accuracy
                loss[class_name]['Precision'] += precision
            except:    
                loss.update({class_name:{'Accuracy':accuracy,'Precision':precision}})
        
        resultsFile.close()
    
    resultsFile = open('Results.txt','a+')
    resultsFile.write('\n************************\n')
    for class_name in loss: #averages all of our class averages
        loss[class_name]['Accuracy'] /= 10
        loss[class_name]['Precision'] /= 10
        resultsFile.write('Accuracy-'+class_name+": "+str(loss[class_name]['Accuracy'])+"\n")
        resultsFile.write('Precision-'+class_name+": "+str(loss[class_name]['Precision'])+"\n")
    resultsFile.write('************************\n')   
    resultsFile.close()

    #average our loss functions
    




def main():

    cleanDiscreteData = ["breast-cancer-wisconsin-cleaned", "glass-cleaned-discrete", "house-votes-84", "iris-cleaned-discrete", "soybean-small-cleaned"]
    scrambledData = ["breast-cancer-wisconsin-cleaned-scrambled", "glass-cleaned-discrete-scrambled", "house-votes-84-scrambled", "iris-cleaned-discrete-scrambled", "soybean-small-cleaned-scrambled"]
    classCol = [10, 10, 0, 4, 35] # the column number of the class - use this to tell our algorithm which column to try to guess

    try:
        os.remove("Results.txt")
    except:
        print("Results File not removed!!!!!!!!!!!")

    
     
    #for every data set and its according randomly shuffled data set
    for i in range(len(classCol)):
        #writes header
        resultsFile = open('Results.txt','a+') 
        resultsFile.write("***************************\n")
        resultsFile.write(cleanDiscreteData[i])
        resultsFile.write("\n***************************\n")
        resultsFile.close()
        file = open('./processed_data/'+ cleanDiscreteData[i] +'.data','r')
        class_column = classCol[i]
        ten_strata, class_counts = stratify(file,class_column)#divide each data pull into ten equally proportioned sections 'ten_strata'
        file.close()
        training(ten_strata, class_column)#preform 10 fold validation for our data

        #repeat for randomized data
        resultsFile = open('Results.txt','a+') 
        resultsFile.write("***************************\n")
        resultsFile.write(scrambledData[i])
        resultsFile.write("\n***************************\n")
        resultsFile.close()
        file = open('./processed_data/'+ scrambledData[i] +'.data','r')
        class_column = classCol[i]
        ten_strata, class_counts = stratify(file,class_column)
        file.close()
        training(ten_strata, class_column)


    

main()