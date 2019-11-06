import pandas as pd

class Data_Processing_Pd:
    #input-> ex: data_processing("abalone", 0, "./data")
    def __init__(self, name, column_class, location):
        self.name = name
        self.column_class = column_class
        self.location = location
        self.df = pd.read_csv(location + "/" + name + ".csv")
        self.sliced = False  
        self.shuffled = False  
        self.column_names_as_numbers = False   
        self.class_at_front = False  

    #names the columns of a df "0","1","2","3","4","5"...
    def name_pd_df_columns(self):
        column_index_names = []
        for i in range (len(self.df.columns)):
            column_index_names.append(str(i))
        self.df.columns = column_index_names
        self.column_names_as_numbers = True

    #prints ENTIRE df
    def print_df(self):
        print(self.df.to_string())
    
    #returns string of ENTIRE df
    def get_df_as_string(self):
        return self.df.to_string()

    def shuffle_rows_df(self):
        self.df = self.df.sample(frac=1)
        self.suffled = True

    #input-> {"string":"number"} ex: {"M":"1", "F":"2", "I":"3"}
    def strings_to_specific_num(self, replacement_dict):
        self.df = self.df.replace(replacement_dict)
    
    #input-> column number (as str or int) to grab all uniqe values from and map to a number
    def unique_column_values_to_num(self, column_num):
        if(self.column_names_as_numbers == False):
            raise Exception ('the columns have not yet been named "0","1","2","3","4","5"...')
        unique_values = self.df[str(column_num)].unique().tolist()
        values = []
        for i in range(len(unique_values)):
            values.append(str(i+1))
        machine = dict(zip(unique_values, values))
        self.df = self.df.replace(machine)

    def write_df_csv(self, location, name):
        if(name == "auto"):
            name = self.name + "_processed"
        else:
            name += ".csv"
        self.df.to_csv(location + "/" + name + ".csv", index= False, header= False)
    
    #pull the class to the front column of the df
    #get a list of the column names
    def pull_classes_front(self):
        column_names = list(self.df.columns)
        column_names.pop(self.column_class)
        column_names.insert(0, str(self.column_class))
        self.df = self.df.reindex(columns = column_names)
        self.class_at_front = True
    
    def slicer(self, sections):   
        self.df = self.df.array_split(self.df, sections)
        self.sliced = True

    #makes all slices in dataframe one unified dataframe
    def concat_df(self):
        if(self.sliced == False):
            pass
        elif (len(self.df)) < 2:
            pass
        else:
            single_dataframe = pd.concat([self.df[0], self.df[1]], sort=False)
            for i in range(len(self.df)-2):
                single_dataframe = pd.concat([single_dataframe, self.df[i+2]], sort=False)
            self.df = single_datafram


if __name__ == "__main__":
    #abalone
    data_aba = Data_Processing_Pd("abalone", 0, "./data")
    data_aba.strings_to_specific_num({"M":"1", "F":"2", "I":"3"})
    data_aba.name_pd_df_columns()
    data_aba.pull_classes_front()
    data_aba.shuffle_rows_df()
    data_aba.write_df_csv("./processed", "auto")

    #car
    data_car = Data_Processing_Pd("car", 6, "./data")
    data_car.strings_to_specific_num({"vhigh":"4", "high":"3", "med":"2", "low":"1", "5more":"5", "more":"6" ,"small":"1", "big":"3", "unacc":"1", "acc":"2", "good":"3", "vgood":"4"})
    data_car.name_pd_df_columns()
    data_car.pull_classes_front()
    data_car.shuffle_rows_df()
    data_car.write_df_csv("./processed", "auto")
    
    #segmentation
    data_img = Data_Processing_Pd("segmentation", 0, "./data")
    data_img.name_pd_df_columns()
    data_img.strings_to_specific_num({"FOLIAGE":"1","PATH":"2","BRICKFACE":"3","GRASS":"4", "SKY":"5", "WINDOW":"6", "CEMENT":"7"})
    data_img.pull_classes_front()    
    data_img.shuffle_rows_df()
    data_img.write_df_csv("./processed", "auto")

    #machine
    data_mach = Data_Processing_Pd("machine", 0, "./data")
    data_mach.name_pd_df_columns()
    data_mach.unique_column_values_to_num(0)
    data_mach.unique_column_values_to_num(1)
    data_mach.pull_classes_front()
    data_mach.shuffle_rows_df()
    data_mach.write_df_csv("./processed", "auto")

    #forestfires
    data_ff = Data_Processing_Pd("forestfires", 12, "./data")
    data_ff.strings_to_specific_num({"jan":"1", "feb":"2", "mar":"3", "apr":"4", "may":"5", "jun":"6", "jul":"7",
    "aug":"8", "sep":"9", "oct":"10", "nov":"11", "dec":"12", "sun":"1", "mon":"2",
    "tue":"3", "wed":"4", "thu":"5", "fri":"6", "sat":"7"})
    data_ff.name_pd_df_columns()
    data_ff.pull_classes_front()
    data_ff.shuffle_rows_df()
    data_ff.write_df_csv("./processed", "auto")    

    #wine
    data_wine = Data_Processing_Pd("wine", 0, "./data")
    data_wine.name_pd_df_columns()
    data_wine.pull_classes_front()
    data_wine.shuffle_rows_df()
    data_wine.write_df_csv("./processed", "auto")
