#abalone        
#class:         [0]
#Real           [1,2,3,4,5,6,7]
#Descrete       [8]

#car        
#class:         [6]
#Real           []
#Descrete       [0,1,2,3,4,5]

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

df = pd.read_csv("./data/wine.csv", index_col=0) 
column_names = ["1","2","3","4","5","6","7","8","9","10","11","12","13"]
df.columns = column_names



dataDesc = df.describe()


for i in range(len(column_names)):
    new_col = pd.cut(df[column_names[i]], bins=[dataDesc.iloc[3][i], dataDesc.iloc[4][i], dataDesc.iloc[5][i], dataDesc.iloc[6][i], dataDesc.iloc[7][i]], labels=["1", "2", "3", "4"])
    df[column_names[i]] = new_col

df.to_csv("./processed/wine_discrete.csv", sep=",", index=True, header=False)
#write data frame out to new file

#columns={1,2,3,4,5,6,7,8,9,10,11,12,13,14}
#




#print(dataDesc.head())
#print(dataDesc.info())
#print(dataDesc.iloc[4][0])

#print(dataDesc)

#col = (df["14.23"])
#print(col)







#print(df.describe())

#print(dataDesc.iloc[4][0])
