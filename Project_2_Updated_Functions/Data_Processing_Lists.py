class Data_Processing_Lists:
    def __init__(self, location, name):
        self.location = location
        self.name = name
        self.sliced = False
        try:
            data = open(location + "/" + name + ".csv", "r")
            self.file_array = data.readlines()
            for i in range (len(self.file_array)):
                #get rid of the "\n" character
                self.file_array[i] = self.file_array[i][:-1]
                self.file_array[i] = self.file_array[i].split(",")
                # for elem in self.file_array[i]:
                #     elem = float(elem)
                for j in range(len(self.file_array[i])):
                    self.file_array[i][j] = float(self.file_array[i][j])
        except FileNotFoundError as fnf_error:
            print(fnf_error)
            return   

    #input-> a 2d array [[line1],[line2]...]
    #return-> a 3d array [[[line1],[line2]], [[line3],[line4]]... ]
    def slicer(self, sections):
        init_len = int(len(self.file_array) / sections)
        new_file_array = []
        for i in range(sections):
            new_file_array.append(self.file_array[i*init_len : init_len*(i+1)])
        self.sliced = True
        #append on remaining lines so we do not lose data
        remaining_lines = len(self.file_array) % sections
        for i in range(remaining_lines):
            #grab from the back of the array (-(0+1) = -1)
            new_file_array[i].append(self.file_array[-(i+1)])
        self.file_array = new_file_array

    def join_array(self):
        new_array = []
        for i in range(len(self.file_array)):
            for j in range(len(self.file_array[i])):
                new_array.append(self.file_array[i][j])
        self.file_array = new_array
            
        

        

        
    