


def catData(fileName):
    try:
        file = open('./processed_data/'+fileName+'.data','r')
    except:
        print("The file didnt open")
        return
    

    lines = file.readlines()
    file.close

    newClassData = []

    #dividing data into 4 classes based on statistical distribution below
    newFile = open('./processed_data/'+fileName+'-discrete','w+')
    
    for line in lines:
        oneLine = line.split(",")
        
        #sepal Length
        # min 4.3, Q2 5.01, mean 5.84, Q3 6.67, Max 7.9
        if line[0] < 5.01:
            newClassData.append("1")
        elif line[0] < 5.84:
            newClassData.append("2")
        elif line[0] < 6.67:
            newClassData.append("3")
        else:
            newClassData.append("4")

        
        #Sepal width
        # min 2.0, Q2 2.62, mean 3.05, Q3 3.48, Max 4.4
        if line[1] < 2.62:
            newClassData.append("1")
        elif line[1] < 3.05:
            newClassData.append("2")
        elif line[1] < 3.48:
            newClassData.append("3")
        else:
            newClassData.append("4")


        #petal length
        # min 1.0, Q2 2, mean 3.76, Q3 5.52, 6.9
        if line[1] < 2:
            newClassData.append("1")
        elif line[1] < 3.76:
            newClassData.append("2")
        elif line[1] < 5.52:
            newClassData.append("3")
        else:
            newClassData.append("4")

        #petal width
        #min .1, Q2 .44, mean 1.2, Q3 1.96, max 2.5
        if line[1] < .44:
            newClassData.append("1")
        elif line[1] < 1.2:
            newClassData.append("2")
        elif line[1] < 1.96:
            newClassData.append("3")
        else:
            newClassData.append("4")


        newFile.write(newClassData+"\n")

        newFile.close()
        




    

def main():
    catData("iris-clean")

if __name__ == "__main__":
    main()