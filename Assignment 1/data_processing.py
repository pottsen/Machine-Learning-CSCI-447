import random

#do not add filetype to filename parameter. This is so it can handle the .data and .names files and create its own
def scramble_data(filename):
    percent_scrambled = 10

    file = open('./data/'+filename+'.data','r')
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
    value_dict = find_value_span('breast-cancer-wisconsin')

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

def find_value_span(filename):
    file = open('./data/'+filename+'.data','r')
    lines = file.readlines()

    values = {}
    line = lines[0].split(",")
    i = 0
    for j in line:
        j=j.replace('\n','')
        values['col'+str(i)] = [j]
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
                values['col'+str(i)].append(k)
            i+=1
    file.close()
    return occurances

scramble_data('breast-cancer-wisconsin')
#print( find_value_span('breast-cancer-wisconsin'))
