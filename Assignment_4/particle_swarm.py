from population_manager import PopulationManager
import random
import numpy as np
import copy
from data_processing import Data_Processing

class particle_swarm(PopulationManager):

    def __init__(self, pop_size, mlp_dims, pBest_coeff, gBest_coeff, inertia_coeff, training_data, test_data):

        PopulationManager.__init__(self, pop_size, mlp_dims)
        self.pBest_C = pBest_coeff
        self.gBest_C = gBest_coeff
        self.pBest_fitness = [0]*pop_size
        self.pBest_fitness_flag = [False]*pop_size
        self.pBest = copy.deepcopy(self.population)
        self.prev_velocity = copy.deepcopy(self.population)
        self.gBest_fitness = 0
        self.gBest = copy.deepcopy(self.population[0])
        self.inertia_C = inertia_coeff
        self.count = 0

        self.training_data_outputs = []
        self.training_data_inputs = []
        for i in training_data:
            actual_class = i[0]
            temp = [0] * mlp_dims[-1]
            temp[int(actual_class-1)] = 1
            self.training_data_inputs.append(i[1:])
            self.training_data_outputs.append(temp)

        self.test_data_outputs = []
        self.test_data_inputs = []
        for i in test_data:
            actual_class = i[0]
            temp = [0] * mlp_dims[-1]
            temp[int(actual_class-1)] = 1
            self.test_data_inputs.append(i[1:])
            self.test_data_outputs.append(temp)

        print(self.test_data_inputs)
        print(self.test_data_outputs)

        PopulationManager.__init__(self, pop_size, mlp_dims)


    def update(self):
        for i in range(len(self.population)):
            # print(i)
            old_pop = self.population[i]
            weights = self.population[i].unzip_neuron()
            gBest_weights = self.gBest.unzip_neuron()
            pBest_weights = self.pBest[i].unzip_neuron()
            prev_velocity = self.prev_velocity[i].unzip_neuron()
            print("pBest updated", self.pBest_fitness_flag[i])

            velocity = self.velocity_calc(prev_velocity, weights, pBest_weights, gBest_weights)
            # print("velocity ", velocity[:10])
            new_weights = []
            for j in range(len(velocity)):
                # print("i",i)
                nw= weights[j] + velocity[j]
                new_weights.append(nw)

            print("weights updated ", new_weights != weights)
            self.prev_velocity[i].rezip_neuron(velocity)
            self.pBest[i].rezip_neuron(pBest_weights)
            self.gBest.rezip_neuron(gBest_weights)
            self.population[i].rezip_neuron(new_weights)
            # print(self.population[i].layers[0].next_weights[0])

            #it is updating based on this
            # print("pop same \n", old_pop.layers[0].next_weights[0], "\n", self.population[i].layers[0].next_weights[0])

        self.count +=1

    def velocity_calc(self, prev_velocity, weights, pBest_weights, gBest_weights):
        #weights = self.population[i].layers[j].next_weights
        velocity = []
        for i in range(len(prev_velocity)):
            v1 = self.inertia_C*prev_velocity[i]
            # v1 = 0
            v2 = self.pBest_C*random.uniform(0,1) * (pBest_weights[i] - weights[i])
            v3 = self.gBest_C * random.uniform(0,1) * (gBest_weights[i] - weights[i])

            if i == 0:
                print("pBest = weights ", pBest_weights == weights)
                # print("PBC ", self.pBest_C)
                print("PBW ", pBest_weights[i])
                print("W", weights[i])
                print("v1 ", v1, "\n v2 ", v2, "\n v3 ", v3)

            v = v1 + v2 + v3
            velocity.append(v)
        return velocity

    def run_PSO(self):
        #iteration count
        iteration = 0
        #calculate fitness
        while iteration < 100:
            # print("iteration ", iteration)
            print("gBest Fitness ", self.gBest_fitness)
            for i in range(len(self.population)):
                fitness = self.population[i].fitness(self.training_data_inputs, self.training_data_outputs)
                # print("Fitness ", fitness)
                #if fitness < pBest reset
                if fitness > self.pBest_fitness[i]:
                    # print("pBest updated")
                    self.pBest[i] = copy.deepcopy(self.population[i])
                    self.pBest_fitness[i] = fitness
                    self.pBest_fitness_flag[i] = True
                else:
                    self.pBest_fitness_flag[i] = False
                #if fitness < gbest reset
                if fitness > self.gBest_fitness:
                    print("gBest updated", self.gBest_fitness)
                    self.gBest = copy.deepcopy(self.population[i])
                    self.gBest_fitness = fitness

            self.update()
            iteration += 1
            print("iteration ", iteration, " fitness ", self.gBest_fitness)

    def run_test(self, test_data):
        


    def f_score(self, guesses): #list of tuples [(actual, guess),(actual,guess)]
        confusion = {} #confusion matrix

        unique_classes = []
        for i in guesses:
            if i[0] not in unique_classes:
                unique_classes.append(i)

        #for each class, initialize the confusion matrix with zeros for that class
        for class_name in unique_classes:
            confusion.update({class_name:{'TP':0,'FP':0,'TN':0,'FN':0}})#class_name is the key for each classes confusion matrix
            #confusion{class:{TP:0,FP:0,TN:0,FN:0}}

        #for each class
        for class_name in unique_classes:
            #for each data point guessed in that class
            for result in guesses: #result[0] is actual class and result[1] is our guess
                if class_name == result[1] and class_name == result[0]: #guess is accurate with what the class actually was
                    value = 'TP'
                if class_name == result[1] and class_name != result[0]: #guessed that a record was part of a class and it wasn't
                    value = 'FP'
                if class_name != result[1] and class_name == result[0]: #guessed that a record was not part of a class and it was
                    value = 'FN'
                if class_name != result[1] and class_name != result[0]: #guess is accurate that the record did not belong to a class
                    value = 'TN'
                confusion[class_name][value] += 1 #increment that classes TP/FP/TN/FN count accordingly

        #calculate our class independent accuracy
        correct = 0
        total = 0
        for result in guesses:
            if(result[0]==result[1]):
                correct+=1
            total+=1
        accuracy = correct/total


        num_of_classes = len(confusion)

        count = 0
        precision = 0
        recall=0
        f1=0
        for class1, matrix in confusion.items():
            TP = matrix['TP']
            TN = matrix['TN']
            FP = matrix['FP']
            FN = matrix['FN']
            if((TP+FP) != 0):
                precision += TP/(TP+FP)
                ptemp = TP/(TP+FP)
            else:
                ptemp = 0
            if((TP+FN) != 0):
                recall += TP/(TP+FN)
                rtemp = TP/(TP+FN)
            else:
                rtemp = 0
            if((ptemp+rtemp)!=0):
                f1 += 2*ptemp*rtemp/(ptemp+rtemp)
            count+=1
        precision = precision/count
        recall = recall/count
        f1 = f1/count

        #f1 = 2*precision*recall/(precision+recall)

        metrics = {'F1': f1, 'Precision':precision, 'Recall':recall, 'Accuracy': accuracy}
        return metrics

if __name__ == "__main__":

    #Prepping data ----------------------------
    #abalone
    data_aba = Data_Processing(["abalone",], [8], {"M":"1", "F":"2", "I":"3"})
    # data_car = Data_Processing(["car",], [6], {"vhigh":"4", "high":"3", "med":"2", "low":"1", "5more":"5", "more":"6","small":"1", "big":"3", "unacc":"1", "acc":"2", "good":"3", "vgood":"4"})
    data_img = Data_Processing(["segmentation",], [0], {"FOLIAGE":"1","PATH":"2","BRICKFACE":"3","GRASS":"4", "SKY":"5", "WINDOW":"6", "CEMENT":"7"})
    # data_mach = Data_Processing(["machine",], [-1], {})
    data_ff = Data_Processing(["forestfires",], [-1],{"jan":"1", "feb":"2", "mar":"3", "apr":"4", "may":"5", "jun":"6","jul":"7", "aug":"8", "sep":"9", "oct":"10", "nov":"11", "dec":"12", "sun":"1", "mon":"2","tue":"3", "wed":"4", "thu":"5", "fri":"6", "sat":"7"})
    data_wine = Data_Processing(["wine",], [-1], {})

    #PROCESS DATA
    data_aba.process_data("./data")
    # data_car.process_data("./data")
    data_img.process_data("./data")
    # data_mach.process_data("./data")
    # data_ff.process_data("./data")
    data_wine.process_data("./data")

    #WRITE DATA
    data_aba.write_data("./processed")
    # data_car.write_data("./processed")
    data_img.write_data("./processed")
    # data_mach.write_data("./processed")
    # data_ff.write_data("./processed")
    data_wine.write_data("./processed")

    #LOAD DATA
    data_aba.load_data("./processed")
    # data_car.load_data("./processed")
    data_img.load_data("./processed")
    # data_mach.load_data("./processed")
    # data_ff.load_data("./processed")
    data_wine.load_data("./processed")



    #triming it down to 10
    data_aba.file_array['abalone'] = data_aba.file_array['abalone']    #slice in to 5
    data_aba.slicer(5, "abalone")

    test_data = data_aba.file_array[0]
    training_data = data_aba.combine(data_aba.file_array[1:])
    #end --------------------------------------

    #tesing alogorithm ------------------------

    # print(len(data_aba.file_array[0][0][1:]))
    # inputs -> (self, pop_size, mlp_dims, pBest_coeff, gBest_coeff, inertia_coeff, training_data, test_data)
    pso = particle_swarm(100, [(len(data_aba.file_array[0][0][1:])),29], 2, 2, 0.4, training_data, test_data)

    pso.run_PSO()





    #end --------------------------------------
