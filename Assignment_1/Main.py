import numpy as np

#Machine Learning: Naive Bayes on 5 data sets
#Peter Ottsen, Bruce Clark, Forest Edwards, Justin McGowen

# TODO:
# Clean data using purely random values - done
# Clean real valued data using data ranges - incomplete
# Seperate data into 10 strata - done
#------10 fold validation-------
# Calculate class probabilities (proportion of one class over the total)- 75% - have counts probabilities are easy from here
# Calculate conditional probabilities of each attribute value - incomplete
# Calculate conditional probabilities of real valued data(data ranges) - incomplete
# Run test data and predict classes
# Find output of two different loss functions based on prediction
#-------------------------------
# randomize data - complete
# repeat 10 fold validation for randomized data
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
    '''
    2 of 10 records:
    [['1000025,5,1,1,1,2,1,3,1,1,2', '1002945,5,4,4,5,7,10,3,2,1,2', '1015425,3,1,1,1,2,2,3,1,1,2', '1016277,6,8,8,1,3,4,3,7,1,2', '1017023,4,1,1,3,2,1,3,1,1,2', '1018099,1,1,1,1,2,10,3,1,1,2', '1018561,2,1,2,1,2,1,3,1,1,2', '1033078,2,1,1,1,2,1,1,1,5,2', '1033078,4,2,1,1,2,1,2,1,1,2', '1035283,1,1,1,1,1,1,3,1,1,2', '1036172,2,1,1,1,2,1,2,1,1,2', '1043999,1,1,1,1,2,3,3,1,1,2', '1048672,4,1,1,1,2,1,2,1,1,2', '1049815,4,1,1,1,2,1,3,1,1,2', '1050718,6,1,1,1,2,1,3,1,1,2', '1056784,3,1,1,1,2,1,2,1,1,2', '1059552,1,1,1,1,2,1,3,1,1,2', '1066373,3,2,1,1,1,1,2,1,1,2', '1066979,5,1,1,1,2,1,2,1,1,2', '1067444,2,1,1,1,2,1,2,1,1,2', '1070935,1,1,3,1,2,1,1,1,1,2', '1070935,3,1,1,1,1,1,2,1,1,2', '1071760,2,1,1,1,2,1,3,1,1,2', '1074610,2,1,1,2,2,1,3,1,1,2', '1075123,3,1,2,1,2,1,2,1,1,2', '1079304,2,1,1,1,2,1,2,1,1,2', '1081791,6,2,1,1,1,1,7,1,1,2', '1096800,6,6,6,9,6,5,7,8,1,2', '1103722,1,1,1,1,2,1,2,1,2,2', '1105524,1,1,1,1,2,1,2,1,1,2', '1106095,4,1,1,3,2,1,3,1,1,2', '1115293,1,1,1,1,2,2,2,1,1,2', '1116192,1,1,1,1,2,1,2,1,1,2', '1117152,4,1,1,1,2,1,3,1,1,2', '1121732,1,1,1,1,2,1,3,2,1,2', '1121919,5,1,3,1,2,1,2,1,1,2', '1124651,1,3,3,2,2,1,7,2,1,2', '1131294,1,1,2,1,2,2,4,2,1,2', '1132347,1,1,4,1,2,1,2,1,1,2', '1133041,5,3,1,2,2,1,2,1,1,2', '1133136,3,1,1,1,2,3,3,1,1,2', '1136142,2,1,1,1,3,1,2,1,1,2', '1137156,2,2,2,1,1,1,7,1,1,2', '1143978,4,1,1,2,2,1,2,1,1,2', '1143978,5,2,1,1,2,1,3,1,1,2', '1017122,8,10,10,8,7,10,9,7,1,4', '1041801,5,3,3,3,2,3,4,4,1,4', '1044572,8,7,5,10,7,9,5,5,4,4', '1047630,7,4,6,4,6,1,4,3,1,4', '1050670,10,7,7,6,4,10,4,1,2,4', '1054590,7,3,2,10,5,10,5,4,4,4', '1054593,10,5,5,3,6,7,7,10,1,4', '1057013,8,4,5,1,2,8,7,3,1,4', '1065726,5,2,3,4,2,7,3,6,1,4', '1072179,10,7,7,3,8,5,7,4,3,4', '1080185,10,10,10,8,6,1,8,9,1,4', '1084584,5,4,4,9,2,10,5,6,1,4', '1091262,2,5,3,3,6,7,7,5,1,4', '1099510,10,4,3,1,3,3,6,5,2,4', '1100524,6,10,10,2,8,10,7,3,3,4', '1102573,5,6,5,6,10,1,3,1,1,4', '1103608,10,10,10,4,8,1,8,10,1,4', '1105257,3,7,7,4,4,9,4,8,1,4', '1106829,7,8,7,2,4,8,3,8,2,4', '1108370,9,5,8,1,2,3,2,1,5,4', '1108449,5,3,3,4,2,4,3,4,1,4', '1110102,10,3,6,2,3,5,4,10,2,4', '1110503,5,5,5,8,10,8,7,3,7,4', '1110524,10,5,5,6,8,8,7,1,1,4'],

    ['1147044,3,1,1,1,2,2,7,1,1,2', '1152331,4,1,1,1,2,1,3,1,1,2', '1155546,2,1,1,2,3,1,2,1,1,2', '1156272,1,1,1,1,2,1,3,1,1,2', '1156948,3,1,1,2,2,1,1,1,1,2', '1157734,4,1,1,1,2,1,3,1,1,2', '1158247,1,1,1,1,2,1,2,1,1,2', '1160476,2,1,1,1,2,1,3,1,1,2', '1164066,1,1,1,1,2,1,3,1,1,2', '1165297,2,1,1,2,2,1,1,1,1,2', '1165790,5,1,1,1,2,1,3,1,1,2', '1167471,4,1,2,1,2,1,3,1,1,2', '1171710,1,1,1,1,2,1,2,3,1,2', '1171795,1,3,1,2,2,2,5,3,2,2', '1173235,3,3,2,1,2,3,3,1,1,2', '1173347,1,1,1,1,2,5,1,1,1,2', '1173347,8,3,3,1,2,2,3,2,1,2', '1173514,1,1,1,1,4,3,1,1,1,2', '1173681,3,2,1,1,2,2,3,1,1,2', '1174057,1,1,2,2,2,1,3,1,1,2', '1174057,4,2,1,1,2,2,3,1,1,2', '1176406,1,1,1,1,2,1,2,1,1,2', '1177027,3,1,1,1,2,1,3,1,1,2', '1177512,1,1,1,1,10,1,1,1,1,2', '1178580,5,1,3,1,2,1,2,1,1,2', '1179818,2,1,1,1,2,1,3,1,1,2', '1180523,3,1,1,1,2,1,2,2,1,2', '1180831,3,1,1,1,3,1,2,1,1,2', '1181356,5,1,1,1,2,2,3,3,1,2', '1182404,4,1,1,1,2,1,2,1,1,2', '1182410,3,1,1,1,2,1,1,1,1,2', '1183240,4,1,2,1,2,1,2,1,1,2', '1183246,1,1,1,1,1,1,2,1,1,2', '1183516,3,1,1,1,2,1,1,1,1,2', '1183911,2,1,1,1,2,1,1,1,1,2', '1184184,1,1,1,1,2,5,1,1,1,2', '1184241,2,1,1,1,2,1,2,1,1,2', '1184840,1,1,3,1,2,1,2,1,1,2', '1185610,1,1,1,1,3,2,2,1,1,2', '1187457,3,1,1,3,8,1,5,8,1,2', '1188472,1,1,1,1,1,1,3,1,1,2', '1190394,4,1,1,1,2,3,1,1,1,2', '1190485,1,1,1,1,2,1,1,1,1,2', '1193091,1,2,2,1,2,1,2,1,1,2', '1193210,2,1,1,1,2,1,3,1,1,2', '1111249,10,6,6,3,4,5,3,6,1,4', '1112209,8,10,10,1,3,6,3,9,1,4', '1113038,8,2,4,1,5,1,5,4,4,4', '1113483,5,2,3,1,6,10,5,1,1,4', '1113906,9,5,5,2,2,2,5,1,1,4', '1115282,5,3,5,5,3,3,4,10,1,4', '1116116,9,10,10,1,10,8,3,3,1,4', '1116132,6,3,4,1,5,2,3,9,1,4', '1116998,10,4,2,1,3,2,4,3,10,4', '1118039,5,3,4,1,8,10,4,9,1,4', '1120559,8,3,8,3,4,9,8,9,8,4', '1123061,6,10,2,8,10,2,7,8,10,4', '1125035,9,4,5,10,6,10,4,8,1,4', '1126417,10,6,4,1,3,4,3,2,3,4', '1147699,3,5,7,8,8,9,7,10,7,4', '1147748,5,10,6,1,10,4,4,10,10,4', '1148278,3,3,6,4,5,8,4,4,1,4', '1148873,3,6,6,6,5,10,6,8,3,4', '1165926,9,6,9,2,10,6,2,9,10,4', '1166630,7,5,6,10,5,10,7,9,4,4', '1166654,10,3,5,1,10,5,3,10,2,4', '1167439,2,3,4,4,2,5,2,5,1,4', '1168359,8,2,3,1,6,3,7,1,1,4', '1168736,10,10,10,10,10,1,8,8,8,4'], ...
    '''
    #print(class_frequency)
    #print (ten_strata)

    return ten_strata, class_frequency

# need table of frequency of attribute value, given class -> F(Aj|Ci)

def main():

    class_column = 10 #which column the class is in
    file = open('./processed_data/breast-cancer-wisconsin-cleaned.data','r')

    ten_strata, class_counts = stratify(file,class_column)

    file.close()
    classAttributeFrequencies = classAttributeFrequency(ten_strata[0], class_column, class_counts)
    print(classAttributeFrequencies)
    classify(['1047630,7,4,6,4,6,1,4,3,1,4'], classAttributeFrequencies)


'''def uniftyData(ten_strata):
    nine_strata = ["","","","","","","","",""]
    test_strata = ["","","","","","","","",""]
    for i in nine_strata:
        for j in ten_strata:
            if j != i:
                nine_strata[i] = nine_strata[i] + ten_strata[j]
            elif j == i:
                test_strata[i] = ten_strata[i]

def classFrequency(class_dict, classColumnLocation, classes):
    #calculate class frequency
    #calculates Q(C=c_i) = #(x is an element of c_i)/Total # of examples

    for i in classes:
        int presentCount = 0
        int count = 0
        int j = 0
        int[] classFreqArray

        for row in class_dict:
            if class_dict[classColumnLocation] = i:
                presentCount += 1
            count +=1

        classFreq = presentCount/count
        classFreqArray[j] = classFreq
        j += 1
'''

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
                attribute_count = str(attribute_probability[key_class][key_attribute][key_value])

                conditional_probability = attribute_count +'/'+ str(class_counts[key_class]) #prints string of probability in fraction form
                #attribute_count = attribute_probability[key_class][key_attribute][key_value]
                #conditional_probability = attribute_count / class_counts[key_class] #prob = occurances of attribute given a class / total occurences of given class
                attribute_probability[key_class][key_attribute][key_value] = conditional_probability

    return attribute_probability

def classify(example, attribute_probability): 
    prob_calc = [] # Stores the probability of the example appearing in the dataset
    for i in example:
        #print("i =", i)
        columns = i.split(",") # Turns the string into a list, separated by commas
        class_name = columns[10]
        #print("class name ", class_name)
        #print("length columns ", range(len(columns)))
        for j in range(len(columns)):
            for key_class in attribute_probability:
                print("key class", key_class)
                #prob_calc[key_class] = class_frequency
                prob_calc = 0.5
                for key_attribute in attribute_probability[key_class]:
                    print("Key attribute: ", key_attribute)
                    for key_value in attribute_probability[key_class][key_attribute]:
                        if columns[j] == str(key_value):
                            print("key value")
                            print(key_value)
                            prob_calc *= float(attribute_probability[key_class][key_attribute][key_value])
    print(prob_calc)
    return prob_calc




main()
