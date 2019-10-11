import pandas as pd
import numpy as np
from Edited_K_Nearest import edited_k_nearest
from K_Nearest import nearest_k_points, concat_df, k_nearest_neighbor
from K_Medoids import k_medoids
from K_Means import k_means


def shuffle_pd_df(data_frames):
    for i in range(len(data_frames)):
        shuffled_data_frame = data_frames[i][1].sample(frac=1)
        data_frames[i][1] = shuffled_data_frame
    return data_frames

#takes single df with number of sections
#returns dataframe[section]
def slicer(sections, dataframe):   
    return np.array_split(dataframe, sections)


#goes through each file(df) and passes to slicer
def slice_pd_df_using_np(sections, data_frames):
    for i in range(len(data_frames)):
        data_frames[i][1] = slicer(sections, data_frames[i][1])

    return data_frames
    

def clean_data(data_frames):
    #figure out what columns have strings
    #figure out what values are in those strings
    #assign different values numerical numbers
    
    #abalone
    abalone = {"M":"1", "F":"2", "I":"3"}
    data_frames[0][1] = data_frames[0][1].replace(abalone)

    #car
    car = {"vhigh":"4", "high":"3", "med":"2", "low":"1", "5more":"5", "more":"6" 
    ,"small":"1", "big":"3"}
    data_frames[1][1] = data_frames[1][1].replace(car)

    #forestfires
    forrestfires = {"jan":"1", "feb":"2", "mar":"3", "apr":"4", "may":"5", "jun":"6", "jul":"7",
    "aug":"8", "sep":"9", "oct":"10", "nov":"11", "dec":"12", "sun":"1", "mon":"2", 
    "tue":"3", "wed":"4", "thu":"5", "fri":"6", "sat":"7"}
    data_frames[2][1] = data_frames[2][1].replace(forrestfires)
    
    # TODO: clean up variables better ??
    #machine
    vendors = "adviser, amdahl, apollo, basf, bti, burroughs, c.r.d, cambex, cdc, dec, dg, formation, four-phase, gould, harris, honeywell, hp, ibm, ipl, magnuson, microdata, nas, ncr, nixdorf, perkin-elmer, prime, siemens, sperry, sratus, wang"
    vendors = vendors.split(", ")
    
    values = []
    for i in range(len(vendors)):
        values.append(str(i+1))

    machine = dict(zip(vendors, values))
    data_frames[3][1] = data_frames[3][1].replace(machine)
    
    unique_values = data_frames[3][1]["1"].unique().tolist()
    values = []
    for i in range(len(unique_values)):
        values.append(str(i+1))
    machine = dict(zip(unique_values, values))
    data_frames[3][1] = data_frames[3][1].replace(machine)
    

    #segmentation
    #data_frames[4][1]
    #no cleaning needed
    
    
    #wine
    #data_frames[5][1]
    #no clearning needed
    
    return data_frames

def normalize(data_frames):
    #data_frames = pd.DataFrame(data_frames)
    for i in range(len(data_frames)):
        data = data_frames[i][1]
        #if each column is a number we could do this:
        #normalized_data = (data-data.min())/(data.max()-data.min())

        #this is to account for first column being class:
        col_num = 0
        for feature in data.columns:
            if(col_num>0):
                data[feature] = pd.to_numeric(data[feature])
                max_val = data[feature].max()
                min_val = data[feature].min()
                data[feature] = (data[feature] - min_val)/(max_val - min_val)
            col_num+=1
        data_frames[i][1] = data


        #for col in range(len(data.iloc[0])-1):

        #    max = np.max(data[str(col+1)])
        #    min = np.min(data[str(col+1)])
        #    print(max)
        #    print(min)
        #    for j in range(len(data)):
        #        data[j][str(col)]= data[j][(col+1)]/np.abs(max-min)



    # for i in range(len(data_frames)):
    #     for j in range(len(data_frames[i][1])):
    #         # data_frames[column]
    #         max = np.max(data_frames[i].loc[:,j])
    #         min = np.min(data_frames[i].loc[:,j])
    #         print(max)
    #        print(min)


    return data_frames

#writes a single data frame to a csv file in the specified path
def write_dataframe_csv(df, location_with_name):
    df.to_csv(location_with_name + ".csv", index= False, header= False)

#takes out dataframes array and passes in the names and the dataframes to write_dataframe_csv
def write_dataframes_csv(dataframes):
    for i in range(len(dataframes)):
        write_dataframe_csv(dataframes[i][1], "./processed/" + dataframes[i][0] + "_processed")


def load_data_from_csv(name_location):
    return pd.read_csv(name_location + ".csv") 

#pulls class_index column to the front of the dataframe
def pull_classes_front(df, class_index):
    #pull the class to the front column of the dataframe
    #get a list of the column names
    column_names = list(df.columns)
    column_names.pop(class_index)
    column_names.insert(0, str(class_index))
    df = df.reindex(columns = column_names)
    return df

# name the columns 0,1,2,3,4,5...
def name_pd_df_columns(df):
    column_index_names = []
    for j in range (len(df.columns)):
        column_index_names.append(str(j))
    df.columns = column_index_names
    return df



def load_data(files, location):

    dataFrames = []    

    for i in range(len(files)):
        #read csv in
        df = pd.read_csv(location+ files[i][0] +".csv") 
        
        # name the columsn 1,2,3,4,5...
        df = name_pd_df_columns(df)
        
        #pull the class to the front column of the dataframe
        df = pull_classes_front(df, files[i][1])
        
        #make data_frames
        info = []
        info.append(files[i][0])
        info.append(df)
        dataFrames.append(info)

    return dataFrames

#writes processed data to csv file
def process_data():

    files = [["abalone", 0],
             ["car", 6],
             ["forestfires", 12],
             ["machine", 0],
             ["segmentation", 0],
             ["wine", 0]] 
    #combine the winedata sets
    df_red = pd.read_csv("./data/" + "winequality-red" + ".csv", sep = ";") 
    df_white = pd.read_csv("./data/" + "winequality-white" + ".csv", sep = ";") 
    df_red = name_pd_df_columns(df_red)
    df_white = name_pd_df_columns(df_white)
    combined_df = concat_df([df_red, df_white])
    write_dataframe_csv(combined_df, "./data/wine")
    


    #load data into pandas--> data_frames[[data_file_Name, dataFrame] , .....]
    data_frames = load_data(files, "./data/")
    
    # shuffle Data
    data_frames = shuffle_pd_df(data_frames)

    # clean Data
    # must turn string values into numbers
    data_frames = clean_data(data_frames)

    #normalize data
    data_frames = normalize(data_frames)

    # Descretize Data
    # With kNN It is not nesessary
    # kNN is a lazy algorithm

    #write the data out into a file
    write_dataframes_csv(data_frames)

           

def cross_validation(folds, k, dataframes, algorithm_name, evaluation_metric):
#dataframes = [db_name, [section1,..,sectionN]]
        #confusion matrix
    guessed_classes = []


    if algorithm_name == 'k-nn':
        print("New Data Set")
        for i in range(folds):
            test_data = dataframes[1].pop(i)
            training_data = concat_df(dataframes[1])

            guessed_classes+=k_nearest_neighbor(k,training_data, test_data)
            dataframes[1].append(test_data)

            
            



    if algorithm_name == 'edited':
        for i in range(folds):
            test_data = dataframes[1].pop(i)
            training_data = concat_df(dataframes[1])
            training_data = edited_k_nearest(k, training_data)
            guessed_classes += k_nearest_neighbor(k,training_data, test_data)
            dataframes[1].append(test_data)


            
            
    
    # if algorithm_name == 'condensed':
    #     for i in range(folds):
    #         test_data = dataframes[1].pop(i)
    #         training_data = concat_df(dataframes[1])
    #         training_data = condensed_k_nearest(k, training_data , .........)
    #         guessed_classes += k_nearest_neighbor(k,training_data, test_data)
    #         dataframes[1].append(test_data)

            

    

    if algorithm_name == 'k-means':
        print("New Data Set")
        for i in range(folds):
            test_data = dataframes[1].pop(i)
            training_data = concat_df(dataframes[1])
            
            #training_data = slicer(4, training_data) # 1/4 of data for this algorithm
            #training_data = shuffle_pd_df(training_data)

            training_data = k_means(k, training_data)
            guessed_classes+=(k_nearest_neighbor(k,training_data, test_data))

            dataframes[1].append(test_data)

    if algorithm_name == 'k-medoids':
        print("New Data Set")
        for i in range(folds):
            test_data = dataframes[1].pop(i)
            training_data = concat_df(dataframes[1])
            
            
            # training_data = shuffle_pd_df(training_data)
            training_data = slicer(4, training_data) # 1/4 of data for this algorithm

            #set medoids to 1/4 data
            medoids = training_data.pop(0)

            #set training data to leftover 3/4 data
            training_data = concat_df(training_data)

            #run PAM-NN
            returned_medoids = k_medoids(medoids, training_data)
            
            #run k-NN with medoids
            guessed_classes += k_nearest_neighbor(k,returned_medoids, test_data)

            dataframes[1].append(test_data)

    #-----------------
    #evaluation metrics for the algorithm's guessed_classes 
    #-----------------
    if evaluation_metric == 'fscore':
        confusion = {} #confusion matrix
        unique_classes = concat_df(dataframes[1])['0'].unique().tolist()
        for class_name in unique_classes:
            confusion.update({class_name:{'TP':0,'FP':0,'TN':0,'FN':0}})#class_name is the key for each classes confusion matrix
        #confusion{class:{TP:0,FP:0,TN:0,FN:0}}

        for class_name in unique_classes:
            for result in guessed_classes: #result[0] is actual class and result[1] is our guess
                #class_name = int(class_name)
                #result[0] = int(result[0])
                #result[1] = int(result[1])
                #print(result)
                if class_name == result[1] and class_name == result[0]: #guess is accurate with what the class actually was
                    value = 'TP'
                if class_name == result[1] and class_name != result[0]: #guessed that a record was part of a class and it wasn't
                    value = 'FP'
                if class_name != result[1] and class_name == result[0]: #guessed that a record was not part of a class and it was
                    value = 'FN'
                if class_name != result[1] and class_name != result[0]: #guess is accurate that the record did not belong to a class
                    value = 'TN'
                confusion[class_name][value] += 1 #increment that classes TP/FP/TN/FN count accordingly
        
        correct = 0
        total = 0
        for result in guessed_classes:
            if(result[0]==result[1]):
                correct+=1
            total+=1
        accuracy = correct/total

        num_of_classes = len(confusion)
        average_cm = {'TP':0,'FP':0,'TN':0,'FN':0}  #average confusion matrix over every class
        print(confusion)

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
            if((TP+FN) != 0):
                recall += TP/(TP+FN)
                rtemp = TP/(TP+FN)
            if((ptemp+rtemp)!=0):
                f1 += 2*ptemp*rtemp/(ptemp+rtemp)
            count+=1
        precision = precision/count
        recall = recall/count
        f1 = f1/count
        
        #f1 = 2*precision*recall/(precision+recall)

        metrics = {'F1': f1, 'Precision':precision, 'Recall':recall, 'Accuracy': accuracy}
        return average_cm, metrics

    if evaluation_metric == 'regression':
        print("regression")

def print_results(matrix, k, file_name, algorithm_name ):
    #matrix = {'F1': f1, 'Precision':precision, 'Recall':recall, 'Accuracy': accuracy}

    results_file = open("./results/" + algorithm_name + "_results.txt", "a+")
    results_file.write(file_name.ljust(10) + " Algorithm_name: " + str(algorithm_name).ljust(20) + "K-value:" + str(k).ljust(10) + "F-score: " + str(matrix['F1']) + " Accuracy: " + str(matrix['Accuracy'])+ " Precision: " + str(matrix['Precision'])+ " Recall: " + str(matrix['Recall'])  + "\n")
    results_file.close()
    print(file_name.ljust(10) + " Algorithm_name: " + str(algorithm_name).ljust(20) + "K-value:" + str(k).ljust(10) + "F-score: " + str(matrix['F1']) + " Accuracy: " + str(matrix['Accuracy']) + " Precision: " + str(matrix['Precision'])+ " Recall: " + str(matrix['Recall'])  +"\n")


def main():

    #processes all data and store in procecessed folder
    #DONT RUN EVERY TIME
    # process_data()

    #load processed data into dataframes 
    files = [["abalone_processed", 0],
             ["car_processed", 0],
             ["segmentation_processed", 0],
             ["machine_processed", 0],
             ["forestfires_processed", 0],
             ["wine_processed", 0]] 
    data_frames = load_data(files, "./processed/")

    #cut the data into ten for validation
    #data_frames = [[(String)name, [[slice1][slice2][slice3][sliceN]]], ...]
    folds = 5
    data_frames= slice_pd_df_using_np(folds, data_frames)
    
    # f,metrics = cross_validation(5, 13, data_frames[4],'k-means')
    
    # cf,fscore = cross_validation(5, 13, data_frames[1],'condensed')
    # print_results(fscore, num, (files[i][0][:-10]), "condensed")

    k = [13, 37, 67]

    for num in k:    
        for file_index in range(len(files)-3):

            #evals = {'F1': f1, 'Precision':precision, 'Recall':recall, 'Accuracy': accuracy}

            cf,evals = cross_validation(folds, num, data_frames[file_index],'k-nn', 'fscore')
            print_results(evals['F1'], num, (files[file_index][0][:-10]), "k-nn")

            cf,evals = cross_validation(folds, num, data_frames[file_index],'edited', 'fscore')
            print_results(evals['F1'], num, (files[file_index][0][:-10]), "edited")            

            cf,evals = cross_validation(folds, num, data_frames[file_index],'condensed', 'fscore')
            print_results(evals['F1'], num, (files[file_index][0][:-10]), "condensed")

    #         cf,evals = cross_validation(folds, num, data_frames[file_index],'k-means')
    #         print_results(evals['F1'], num, (files[file_index][0][:-10]), "k-means")

            # if file_index > 2:
            #     cf,evals = cross_validation(folds, num, data_frames[file_index],'k-medoids')
            #     print_results(evals, num, (files[file_index][0][:-10]), "k-medoids")
        
            

    
    
    

# # Steps
# # 1. reprocess data


# # 2. Run k-NN all data sets


# # 3. Run E-NN on Abalone, Car, and Machine


# # 4. Run C-NN on Abalone, Car, and Machine


# # 5. Run k-Means on E-NN from Abalone, Car, and Image


# # 6. Run PAM-NN on E-NN from Abalone, Car, and Image


# # 7. Run K-means with 1/4n as the means size for Comp HW, Fire, Wine


# # 8. Run PAM-NN with 1/4n  as the medoids size Comp HW, Fire, Wine

    

if __name__ == "__main__":
    main()
