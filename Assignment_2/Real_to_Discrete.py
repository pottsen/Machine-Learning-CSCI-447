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

df = pd.read_csv("./data/wine.csv") 
