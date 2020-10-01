"""
Methods used to process and prepared data for the algorithms
"""

import re

class Data_Processing:
    def __init__(self, names, class_col, cat_to_numeric):
        self.names = names           #names is a list of file names without extentions of the data files
        self.class_col = class_col
        self.file_array = {}
        self.cat_to_numeric = cat_to_numeric  #a dictionary of categorical keys to be replaced with numeric values

    def process_with_strings(self,location):
        for name in range(len(self.names)):

            raw_txt = ''
            data = open(location + "/" + self.names[name] + ".csv", "r")
            for i in data:
                raw_txt+=i
            data.close()

            raw_txt = raw_txt[:-1]#strip off trailing \n
            points_prime = raw_txt.split('\n')

            points = []*len(points_prime)

            for i in range(len(points_prime)):
                tmp = points_prime[i].split(',')
                first_col = tmp[0]
                tmp[0] = tmp[self.class_col[name]]
                tmp[self.class_col[name]] = first_col
                points.append([i for i in tmp])

            self.file_array.update({self.names[name]:points})
    def process_data(self, location):
        for name in range(len(self.names)):

            raw_txt = ''
            data = open(location + "/" + self.names[name] + ".csv", "r")
            for i in data:
                raw_txt+=i
            data.close()

            for key in self.cat_to_numeric:
                raw_txt = re.sub(r"\n"+key+",","\n"+self.cat_to_numeric[key]+",",raw_txt)
                raw_txt = re.sub(r"^"+key+",",self.cat_to_numeric[key]+",",raw_txt)
                raw_txt = re.sub(r","+key+"\n",","+self.cat_to_numeric[key]+"\n",raw_txt)
                raw_txt = re.sub(r","+key+"$",","+self.cat_to_numeric[key],raw_txt)
                raw_txt = raw_txt.replace(","+key+",",","+self.cat_to_numeric[key]+",")
                raw_txt = raw_txt.replace(","+key+",",","+self.cat_to_numeric[key]+",")
            if(raw_txt[-1]=='\n'):
                raw_txt = raw_txt[:-1]#strip off trailing \n
            points_prime = raw_txt.split('\n')

            points = []*len(points_prime)

            for i in range(len(points_prime)):
                tmp = points_prime[i].split(',')
                first_col = tmp[0]
                tmp[0] = tmp[self.class_col[name]]
                tmp[self.class_col[name]] = first_col
                points.append([float(i) for i in tmp])

            self.file_array.update({self.names[name]:points})

    def load_data(self, location):
        for name in self.names:
            file = open(location + "/" + name + "_processed.csv", "r")
            points = []
            for data in file:
                temp = data.strip("\n").split(",")
                points.append([float(i) for i in temp])
            file.close()

            self.file_array.update({name:points})

    def write_data(self, location):
        for name in self.file_array:
            file = open(location + "/" + name + "_processed.csv", "w+")
            for data in self.file_array[name]:
                string = ",".join([str(i) for i in data]) + "\n"
                file.write(string)
            file.close()

    #input-> a 2d array [[line1],[line2]...]
    #return-> a 3d array [[[line1],[line2]], [[line3],[line4]]... ]
    def slicer(self, sections, name):
        init_len = int(len(self.file_array[name]) / sections)
        new_file_array = []
        for i in range(sections):
            new_file_array.append(self.file_array[name][i*init_len : init_len*(i+1)])
        self.sliced = True
        #append on remaining lines so we do not lose data
        remaining_lines = len(self.file_array[name]) % sections
        for i in range(remaining_lines):
            #grab from the back of the array (-(0+1) = -1)
            new_file_array[i].append(self.file_array[name][-(i+1)])
        self.file_array = new_file_array

    #input-> a 3d array [[[line1],[line2]], [[line3],[line4]]... ]
    #return-> a 2d array [[line1],[line2]...]
    def combine(self, file_array):
        new_file_array = []
        for i in file_array:
            for j in i:
                new_file_array.append(j)
        return new_file_array

    def col_namespaces(self):
        namespaces = {}

        for set in self.file_array:
            data = self.file_array[set]
            ns = []
            for x in range(len(data[0])):
                ns.append([])
            namespaces.update({set:ns})
            for point in data:
                for col in range(len(point)):
                    if(point[col] not in namespaces[set][col] and not isnumber(point[col])):
                        namespaces[set][col].append(point[col])

        print(namespaces)

def isnumber(data):
    data = str(data)
    num_re = r"-?\d*\.?\d*"

    search = re.search(num_re,data)[0]

    return search == data
