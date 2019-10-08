import pandas as pd
import numpy as np
from Edited_K_Nearest import edited_k_nearest
from K_Nearest import nearest_k_points, concat_df, k_nearest_neighbor


def shuffle_pd_df(data_frames):
    for i in range(len(data_frames)):
        shuffled_data_frame = data_frames[i][1].sample(frac=1)
        data_frames[i][1] = shuffled_data_frame
    return data_frames

#takes single df with number of sections
#returns dataframe[section]
def slicer(sections, dataframe):   
    return np.array_split(dataframe, 10)


#goes through each file(df) and passes to slicer
def slice_pd_df_using_np(sections, data_frames):
    for i in range(len(data_frames)):
        data_frames[i][1]

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
    
    # TODO: use new wine data
    #wine
    #data_frames[5][1]
    #no clearning needed
    
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

# name the columns 1,2,3,4,5...
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

    # Descretize Data
    # With kNN It is not nesessary
    # kNN is a lazy algorithm

    #write the data out into a file
    write_dataframes_csv(data_frames)

    



def cross_validation(k, dataframes, algorithm_name):
    folds = 10
    if algorithm_name == 'k-nn':
        for i in range(folds):
            # print(dataframes[1])
            test_data = dataframes[1].pop(i)
            #print(test_data)
            training_data = concat_df(dataframes[1])

            guessed_classes = k_nearest_neighbor(k,training_data, test_data)
            print(guessed_classes)

            #TODO Loss functions here
    if algorithm_name == 'condensed':
        for i in range(folds):
            test_data = dataframes[1].pop(i)
            training_data = concat_df(dataframes[1])

            #guessed_classes = Condensed_k_nearest(k,training_data, test_data)

            #TODO Loss functions here

    if algorithm_name == 'edited':
        for i in range(folds):
            test_data = dataframes[1].pop(i)
            training_data = concat_df(dataframes[1])
            training_data = edited_k_nearest(k, training_data)
            guessed_classes = k_nearest_neighbor(k,training_data, test_data)
            



def main():

    #processes all data and store in procecessed folder
    #DONT RUN EVERY TIME
    process_data()

    #load processed data into dataframes 
    files = [["abalone_processed", 0],
             ["car_processed", 0],
             ["forestfires_processed", 0],
             ["machine_processed", 0],
             ["segmentation_processed", 0],
             ["wine_processed", 0]] 
    data_frames = load_data(files, "./processed/")

    #cut the data into ten for validation
    #data_frames = [[(String)name, [[slice1][slice2][slice3][sliceN]]], ...]
    number_of_sections = 10
    data_frames = slice_pd_df_using_np(number_of_sections, data_frames)
    return data_frames


    
    
    
    #perform the nearest neighbor algorithm
    nearest_k_points(1, data_frames[5][1][0], data_frames[5][1][0].iloc[0])
    #cross_validation(67,data_frames[0],'k-nn')


    #Test EditedK_Neatest
    #cross_validation(13, data_frames[4] , "edited")


    


if __name__ == "__main__":
    main()
    

