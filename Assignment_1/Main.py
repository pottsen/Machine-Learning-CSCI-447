#Machine Learning: Naive Bayes on 5 data sets
#Peter Ottsen, Bruce Clark, Forest Edwards, Justin McGowen

# TODO:
# Clean data using purely random values - done
# Clean real valued data using data ranges - done
# Seperate data into 10 strata - done
#------10 fold validation-------
# Calculate class probabilities (proportion of one class over the total)- done
# Calculate conditional probabilities of each attribute value - done
# Calculate conditional probabilities of real valued data(data ranges) - done
# Run test data and predict classes - in progress
# Find output of two different loss functions based on prediction
#-------------------------------
# randomize data - complete
# repeat 10 fold validation for randomized data - in progress
# ----------Paper----------
# 'Develop hypothesis' for each data-set
# do paper!
# create video of code functionality

#optional:
#add attribute header names to dataset


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
        strata_subsize = int(class_frequency[key]/10)
        for i in range(10):
            start_range = i * strata_subsize
            end_range = i * strata_subsize + strata_subsize
            records = class_data[key].split('\n')
            ten_strata[i]+=(records[start_range:end_range])
    
    #print(class_frequency)
    #print (ten_strata)

    return ten_strata, class_frequency

# need table of frequency of attribute value, given class -> F(Aj|Ci)

def main():

    class_column = 10 #which column the class is in
    
    file = open('./processed_data/glass-cleaned-scrambled.data','r')
    #file = open('./data/house-votes-84.data','r')


    ten_strata, class_counts = stratify(file,class_column)

    file.close()
    #print(ten_strata)
    training = []
    #ten fold validation
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

        resultsFile = open('Results.txt','a')

        confusion = {}
        for class_name in classAttributeFrequencies:
            confusion.update({class_name:{'TP':0,'FP':0,'TN':0,'FN':0}})
        #confusion{class:{TP:0,FP:0,TN:0,FN:0}}
        for i in ten_strata[j]:
            actual_class = i.split(',')[class_column]
            guess = classify(i, classAttributeFrequencies, class_column)
            #print(guess)            
            for class_name in classAttributeFrequencies:
                if class_name == guess and class_name == actual_class:
                    value = 'TP'
                if class_name == guess and class_name != actual_class:
                    value = 'FP'
                if class_name != guess and class_name == actual_class:
                    value = 'FN'
                if class_name != guess and class_name != actual_class:
                    value = 'TN'
                confusion[class_name][value] += 1
        
        print(confusion)
        

        for key,val in confusion.items():
            output = str(((key, " = ", val)))
            resultsFile.write(output)
            resultsFile.write("\n")


        
        for class_name in classAttributeFrequencies:
            total = confusion[class_name]['TP'] + confusion[class_name]['FP'] + confusion[class_name]['TN'] + confusion[class_name]['FN']
            accuracy = (confusion[class_name]['TP'] + confusion[class_name]['TN'])/ total
            #print('Accuracy-'+class_name+": "+str(accuracy))
            resultsFile.write('Accuracy-'+class_name+": "+str(accuracy))
            positives = confusion[class_name]['TP'] + confusion[class_name]['FP']
            try:
                precision = confusion[class_name]['TP'] / positives
            except:
                precision = 'INF'
            #print('Precision-'+class_name+": "+str(precision))
            resultsFile.write('Precision-'+class_name+": "+str(precision))

        

#Justin: I think I have a pretty good idea of what attribute probabilities will look like
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
        # classes = []
        # if classes = []:
        #     classes = class_name
        # for i in classes:


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

#returns the probability that an example is of a class...
def classify(example, attribute_probability, class_col): 
    prob_calc = {} # Stores the probability of the example appearing in the dataset - probability:class
    columns = example.split(",") # Turns the string into a list, separated by commas
    #class_name = columns[class_col]
    for key_class in attribute_probability:
        #for j in range(len(columns)):
            #print("key class", key_class)
            #prob_calc[key_class] = class_frequency
        class_prob = 1 #class frequency
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

main()