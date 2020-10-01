import pandas as pd
import numpy as np


def catDataGlass(fileName):
    '''
    Attribute:   Min     Max      Mean     SD      Correlation with class
    2. RI:       1.5112  1.5339   1.5184  0.0030  -0.1642
    3. Na:      10.73   17.38    13.4079  0.8166   0.5030
    4. Mg:       0       4.49     2.6845  1.4424  -0.7447
    5. Al:       0.29    3.5      1.4449  0.4993   0.5988
    6. Si:      69.81   75.41    72.6509  0.7745   0.1515
    7. K:        0       6.21     0.4971  0.6522  -0.0100
    8. Ca:       5.43   16.19     8.9570  1.4232   0.0007
    9. Ba:       0       3.15     0.1750  0.4972   0.5751
    10. Fe:       0       0.51     0.0570  0.0974  -0.1879
    '''

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
        newIntOneLine = []
        for i in range (len(oneLine)):
            newIntOneLine.append(float(oneLine[i]))

        oneLine = newIntOneLine

        #get class
        className = oneLine[10]
        newClassData.append(oneLine[0])

        #RI
        if oneLine[1] < 1.5154:
            newClassData.append(1)
        elif oneLine[1] < 1.5184:
            newClassData.append(2)
        elif oneLine[1] < 1.5214:
            newClassData.append(3)
        else:
            newClassData.append(4)


        #Na
        if oneLine[2] < 12.5913:
            newClassData.append(1)
        elif oneLine[2] < 13.4079:
            newClassData.append(2)
        elif oneLine[2] < 14.2245:
            newClassData.append(3)
        else:
            newClassData.append(4)

        #Mg
        if oneLine[3] < 1.2421:
            newClassData.append(1)
        elif oneLine[3] < 2.6845:
            newClassData.append(2)
        elif oneLine[3] < 4.1269:
            newClassData.append(3)
        else:
            newClassData.append(4)

        #Al
        if oneLine[4] < .9456:
            newClassData.append(1)
        elif oneLine[4] < 1.449:
            newClassData.append(2)
        elif oneLine[4] < 1.9442:
            newClassData.append(3)
        else:
            newClassData.append(4)

        #Si
        if oneLine[5] < 71.8764:
            newClassData.append(1)
        elif oneLine[5] < 72.6509:
            newClassData.append(2)
        elif oneLine[5] < 73.4354:
            newClassData.append(3)
        else:
            newClassData.append(4)

        #k
        if oneLine[6] < -0.1551:
            newClassData.append(1)
        elif oneLine[6] < 0.49:
            newClassData.append(2)
        elif oneLine[6] < 1.1493:
            newClassData.append(3)
        else:
            newClassData.append(4)

        #Ca
        if oneLine[7] < 7.5338:
            newClassData.append(1)
        elif oneLine[7] < 8.9570:
            newClassData.append(2)
        elif oneLine[7] < 10.3802:
            newClassData.append(3)
        else:
            newClassData.append(4)


        #Ba
        if oneLine[8] < -.3222:
            newClassData.append(1)
        elif oneLine[8] < .1750:
            newClassData.append(2)
        elif oneLine[8] < .6722:
            newClassData.append(3)
        else:
            newClassData.append(4)


        #Fe
        if oneLine[9] < -.453:
            newClassData.append(1)
        elif oneLine[9] < .0570:
            newClassData.append(2)
        elif oneLine[9] < .567:
            newClassData.append(3)
        else:
            newClassData.append(4)

        newClassData.append(className)

        for i in range(len(newClassData)):
            if i == (len(newClassData)-1):
                newFile.write(str(newClassData[i]))
            else:
                newFile.write(str(newClassData[i]) +", ")
        newFile.write("\n")
        
        
    newFile.close()
    



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
    catDataGlass("glass-cleaned")
    print("finished")

    

if __name__ == "__main__":
    main()