#abalone        
#class:         [0]
#Real           [1,2,3,4,5,6,7]
#Descrete       [8]

#car        
#class:         [6]
#Real           []
#Descrete       []

#forestfires
#class:         [12] - this is a real value
#Real           [4,5,6,7,8,9,10,11,12]
#Discrete       [0,1,2,3,]               

#Machine
#class:         not sure which is the attribute we need to predict
#Real           [2,3,4,5,6,7,8,9]
#Discrete       [0,1]

#Segmentation
#class:         [0]
#Real           [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
#Discrete       []

#Wine
#class:         [0]
#Real           [1,2,3,4,5,6,7,8,9,10,11,12,13]
#Discrete       []

import pandas as pd


def getNamesofColumns(df):
    number_of_columns = []
    for i in range(len(df.columns)):
        number_of_columns.append(str(i))
    return number_of_columns
        


def descretize(list_of_info):

    df = pd.read_csv("./data/"+ list_of_info[0] +".csv", index_col=list_of_info[1]) 

    column_names = getNamesofColumns(df)
    df.columns = column_names

    dataDesc = df.describe()

    

    # for i in range(len(column_names)):
    #     new_col = pd.cut(df[column_names[i]], bins=[dataDesc.iloc[3][i], dataDesc.iloc[4][i], dataDesc.iloc[5][i], dataDesc.iloc[6][i], dataDesc.iloc[7][i]], labels=["1", "2", "3", "4"])
    #     df[column_names[i]] = new_col

    print(dataDesc)

    for col in list_of_info[2]:
        #if the bin are the same value, decrease the number of bins and labels
        new_col = pd.cut(df[column_names[col]], bins=[dataDesc.iloc[3][col], dataDesc.iloc[4][col], dataDesc.iloc[5][col], dataDesc.iloc[6][col], dataDesc.iloc[7][col]], labels=["1", "2", "3", "4"])
        df[column_names[col]] = new_col
        #dataDesc.iloc[3][col], dataDesc.iloc[4][col], dataDesc.iloc[5][col], dataDesc.iloc[6][col], dataDesc.iloc[7][col]

    # df.to_csv("./processed/"+ list_of_info[0] +"_discrete.csv", sep=",", index=True, header=False)


def main():
    #files = [["filename" , column_of_class, [discrete_values_columns]], ......]
    files = [["abalone", 0, [1,2,3,4,5,6,7]],
             #["car", 6, []],
             ["forestfires", 12, [0,1,2,3,12]],
             #["machine", 0, [0,1]],
             ["segmentation", 0, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]],
             ["wine", 0, [1,2,3,4,5,6,7,8,9,10,11,12,13]]] 
    
    for i in range(len(files)):
        descretize(files[i])
        print(i)
    


if __name__ == "__main__":
    main()
