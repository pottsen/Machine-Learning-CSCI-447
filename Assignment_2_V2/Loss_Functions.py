class Loss_Functions:
    #input-> [[actual, guess, index], ... ]
    def __init__(self, results):
        self.results = results
        self.confusion = False
    
    def accuracy(self):
        accurate = 0
        inaccurate = 0
        for i in range(len(self.results)):
            if self.results[i][0] == self.results[i][1]:
                accurate += 1
            else:
                inaccurate += 1
        return accurate / (accurate + inaccurate)

    def confusion_matrix_generator(self):
        #find unique classes
        unique_class = []
        for i in range(len(self.results)):
            if self.results[i][0] not in unique_class:
                unique_class.append(self.results[i][0])

        confusion_matrix = {'TP':0,'FP':0,'TN':0,'FN':0}

        for class_name in unique_class:
            for i in range(len(self.results)):
                if class_name == self.results[i][1] and class_name == self.results[i][0]: #guess is accurate with what the class actually was
                    confusion_matrix["TP"] += 1
                elif class_name == self.results[i][1] and class_name != self.results[i][0]: #guessed that a record was part of a class and it wasn't
                    confusion_matrix["FP"] += 1
                elif class_name != self.results[i][1] and class_name == self.results[i][0]: #guessed that a record was not part of a class and it was
                    confusion_matrix["FN"] += 1
                elif class_name != self.results[i][1] and class_name != self.results[i][0]: #guess is accurate that the record did not belong to a class
                    confusion_matrix["TN"] += 1

        self.confusion_matrix = confusion_matrix
        self.confusion = True

    def f_score(self):
        if (self.confusion == False):
            raise Exception ("Ummm, call confusion_matrix() before calling f_score()")
        TP = self.confusion_matrix["TP"]
        FP = self.confusion_matrix["FP"]
        FN = self.confusion_matrix["FN"]
        TN = self.confusion_matrix["TN"]

        try:
            precision = TP/(TP+FP)
        except:
            precision = 0
        try:
            recall = TP/(TP+FN)
        except:
            recall = 0
        try:
            print("fscore")
            return 2*((precision*recall)/(precision+recall))
        except:
            raise Exception ("precision: " + precision + " or recall: " + recall + " add to 0.")

        
            






