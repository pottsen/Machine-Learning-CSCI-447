import pandas as pd
import numpy as np


def catDataGlass():
    try:
        glass = pd.read_csv("glass-cleaned.csv")
    
    except:
        print("nope")
    
    #dfm = pd.DataFrame(glass['data'])
    #dfm.columns = glass['measurments']
    #print(dfm.shape)  
    #print(dfm.index)

    



def catDataIris(fileName):
    try:
        file = open('./processed_data/'+fileName+'.data','r')
    except:
        print("The file didnt open")
        return
    

    lines = file.readlines()
    file.close

    #dividing data into 4 classes based on statistical distribution below
    newFile = open('./processed_data/'+fileName+'-discrete.data','w+')
    
    for line in lines:
        newClassData = []
        oneLine = line.split(",")
        
        #change to a list of numbers
        #-1 dont want to change class to float
        newIntOneLine = []
        for i in range (len(oneLine)-1):
            newIntOneLine.append(float(oneLine[i]))
        className = oneLine[len(oneLine)-1][0:-2]
        oneLine = newIntOneLine


        #sepal Length
        # min 4.3, Q2 5.01, mean 5.84, Q3 6.67, Max 7.9
        if oneLine[0] < 5.01:
            newClassData.append(1)
        elif oneLine[0] < 5.84:
            newClassData.append(2)
        elif oneLine[0] < 6.67:
            newClassData.append(3)
        else:
            newClassData.append(4)

        #Sepal width
        # min 2.0, Q2 2.62, mean 3.05, Q3 3.48, Max 4.4
        if oneLine[1] < 2.62:
            newClassData.append(1)
        elif oneLine[1] < 3.05:
            newClassData.append(2)
        elif oneLine[1] < 3.48:
            newClassData.append(3)
        else:
            newClassData.append(4)


        #petal length
        # min 1.0, Q2 2, mean 3.76, Q3 5.52, 6.9
        if oneLine[1] < 2:
            newClassData.append(1)
        elif oneLine[1] < 3.76:
            newClassData.append(2)
        elif oneLine[1] < 5.52:
            newClassData.append(3)
        else:
            newClassData.append(4)

        #petal width
        #min .1, Q2 .44, mean 1.2, Q3 1.96, max 2.5
        if oneLine[1] < .44:
            newClassData.append(1)
        elif oneLine[1] < 1.2:
            newClassData.append(2)
        elif oneLine[1] < 1.96:
            newClassData.append(3)
        else:
            newClassData.append(4)

        #append the class name
        newClassData.append(className)

        #write new category to file 
        for i in range (len(newClassData)):
            if i == (len(newClassData)-1):
                newFile.write(str(newClassData[i]))
            else:
                newFile.write((str(newClassData[i]) + ","))
        
        newFile.write("\n")
        
    newFile.close()
        
    

def main():
    #catDataIris("iris-cleaned")
    catDataGlass()
    print("finished")

    

if __name__ == "__main__":
    main()