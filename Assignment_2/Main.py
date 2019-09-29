import pandas as pd
import numpy as np
from K_Nearest import nearest_k_points

def shuffle_pd_df(data_frames):
    for i in range(len(data_frames)):
        shuffled_data_frame = data_frames[i][1].sample(frac=1)
        data_frames[i][1] = shuffled_data_frame
    return data_frames



def slice_pd_df_using_np(sections, data_frames):
    
    for i in range(len(data_frames)):
        valid = False
        while(valid == False):
            new_df = np.array_split(data_frames[i][1], 10)
            valid = True
            data_frames[i][1] = new_df
            

    return data_frames
    

def clean_data(data_frames):
    #figure out what columns have strings
    #figure out what values are in those strings
    #assign different values numerical numbers
    
    #abalone
    abalone = {"M":"1", "F":"2", "I":"3"}
    data_frames[0][1] = data_frames[0][1].replace(abalone)

    #car
    car = {"v-high":"4", "high":"3", "med":"2", "low":"1", "5-more":"5", "more":"6" 
    ,"small":"1", "big":"3"}
    data_frames[1][1] = data_frames[1][1].replace(car)

    #forestfires
    forrestfires = {"jan":"1", "feb":"2", "mar":"3", "apr":"4", "may":"5", "jun":"6", "jul":"7",
    "aug":"8", "sep":"9", "oct":"10", "nov":"11", "dec":"12", "sun":"1", "mon":"2", 
    "tue":"3", "wed":"4", "thu":"5", "fri":"6", "sat":"7"}
    data_frames[2][1] = data_frames[2][1].replace(forrestfires)
    

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


def load_data():

    files = [["abalone", 0],
             ["car", 6],
             ["forestfires", 12],
             ["machine", 0],
             ["segmentation", 0],
             ["wine", 0]] 

    dataFrames = []

    

    for i in range(len(files)):
        #read csv in
        df = pd.read_csv("./data/"+ files[i][0] +".csv") 
        
        # name the columsn 1,2,3,4,5...
        column_index_names = []
        for j in range (len(df.columns)):
            column_index_names.append(str(j))
        df.columns = column_index_names
        
        #pull the class to the front column of the dataframe
        class_num = files[i][1]
        column_index_names.pop(class_num)
        column_index_names.insert(0, str(class_num))
        df = df.reindex(columns = column_index_names)
        
        info = []
        info.append(files[i][0])
        info.append(df)
        dataFrames.append(info)

    return dataFrames
    



def main():
    
    #load data into pandas--> data_frames[[data_file_Name, dataFrame] , .....]
    data_frames = load_data()
    

    # shuffle Data
    data_frames = shuffle_pd_df(data_frames)

    # clean Data
    # must turn string values into numbers
    clean_data(data_frames)

    # Descretize Data
    # With kNN It is not nesessary
    # kNN is a lazy algorithm

    #cut the data into ten for validation
    #divide_data(10)
    #data_frames = [[(String)name, [slice1][slice2][slice3][sliceN]], ...]
    number_of_sections = 10
    data_frames = slice_pd_df_using_np(number_of_sections, data_frames)
    
    #perform the nearest neighbor algorithm
    #print(data_frames[5][1][0])
    nearest_k_points(3, data_frames[5][1][0])




if __name__ == "__main__":
    main()
