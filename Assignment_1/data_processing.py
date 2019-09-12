import random

#do not add filetype to filename parameter. This is so it can handle the .data and .names files and create its own
def scramble_data(filename):
    percent_scrambled = 10

    try:
        file = open('./data/'+filename+'.data','r')
    except:
        file = open('./processed_data/'+filename+'.data','r')
    processed_file = open('./processed_data/'+filename+'-scrambled.data','w+')
    lines = file.readlines()
    file.close()
    file_length = len(lines)
    scrambled_records=[]
    i=0
    while i<= (file_length/percent_scrambled):
        rand = random.randint(1,file_length)
        try:
            val = scrambled_records.index(rand)
        except ValueError:
            scrambled_records.append(rand)
            i+=1
    value_dict = find_value_span(filename)

    i = 0
    for j in lines:
        try:
            val = scrambled_records.index(i+1)
            #scramble rows here
            val = ""
            for k in range(len(value_dict)):
                val += random.choice(value_dict['col'+str(k)]) + ","
            processed_file.write(val[:-1]+"\n")
        except ValueError:
            processed_file.write(j)
        i+=1
    processed_file.close()

def find_value_span(filename): #looks through data and finds options for data and puts options in values
    try:
        file = open('./data/'+filename+'.data','r')
    except:
        file = open('./processed_data/'+filename+'.data','r')

    lines = file.readlines()

    values = {}
    line = lines[0].split(",")
    i = 0
    for j in line:  #for every comma seperated value, add a key for that column (col#)
        j=j.replace('\n','')
        values['col'+str(i)] = []
        i+=1
    for j in lines:
        #processed_file.write(j)
        line = j.split(",")
        i = 0
        for k in line:
            k=k.replace('\n','')
            try:
                values['col'+str(i)].index(k)
            except ValueError:
                if k !='?':
                    values['col'+str(i)].append(k)
            i+=1
    file.close()
    return values

def clean_data(filename):
    file = open('./data/'+filename+'.data','r')
    lines = file.readlines()
    file.close()

    value_dict = find_value_span(filename)

    clean = open('./processed_data/'+filename+'-cleaned.data', 'w+')
    for i in lines:
        line = i.split(",")
        for j in range(len(line)):
            if line[j] == "?":
                line[j] = random.choice(value_dict['col'+str(j)])
            elif(line[j] == "?\n"):
                line[j] = random.choice(value_dict['col'+str(j)]) +'\n'
        lines=""
        for j in line:
            lines+=j+","
        clean.write(lines[:-1])
    clean.close()


    
clean_data('breast-cancer-wisconsin')
scramble_data('breast-cancer-wisconsin-cleaned')
clean_data('glass')
scramble_data('glass-cleaned')
#clean_data('house-votes-84')
scramble_data('house-votes-84')
clean_data('iris')
scramble_data('iris-cleaned-discrete')
clean_data('soybean-small')
scramble_data('soybean-small-cleaned')
