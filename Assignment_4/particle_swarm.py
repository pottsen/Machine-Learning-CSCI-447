"""
Particle Swarm Algorithm
Contains the PSO algorithm and associated code to run 5 fold validation on it.

"""



from population_manager import PopulationManager
import random
import numpy as np
import copy
import operator
from data_processing import Data_Processing
from evaluations import  f_score, mse

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
            if mlp_dims[-1] > 1:
                temp = [0] * mlp_dims[-1]
                temp[int(actual_class-1)] = 1
                self.training_data_inputs.append(i[1:])
                self.training_data_outputs.append(temp)
            else:
                self.training_data_inputs.append(i[1:])
                # print("reached here")
                self.training_data_outputs.append(actual_class)

        self.test_data_outputs = []
        self.test_data_inputs = []
        for i in test_data:
            actual_class = i[0]
            if mlp_dims[-1] > 1:
                temp = [0] * mlp_dims[-1]
                temp[int(actual_class-1)] = 1
                self.test_data_inputs.append(i[1:])
                self.test_data_outputs.append(temp)
            else:
                self.test_data_inputs.append(i[1:])
                self.test_data_outputs.append(i[0])

        PopulationManager.__init__(self, pop_size, mlp_dims)


    def update(self):
        for i in range(len(self.population)):
            weights = self.population[i].unzip_neuron()
            gBest_weights = self.gBest.unzip_neuron()
            pBest_weights = self.pBest[i].unzip_neuron()
            prev_velocity = self.prev_velocity[i].unzip_neuron()
            # print("pBest updated", self.pBest_fitness_flag[i])

            velocity = self.velocity_calc(prev_velocity, weights, pBest_weights, gBest_weights)
            new_weights = []
            for j in range(len(velocity)):
                nw= weights[j] + velocity[j]
                new_weights.append(nw)

            # print("weights updated ", new_weights != weights)
            self.prev_velocity[i].rezip_neuron(velocity)
            self.pBest[i].rezip_neuron(pBest_weights)
            self.gBest.rezip_neuron(gBest_weights)
            self.population[i].rezip_neuron(new_weights)


        self.count +=1

    def velocity_calc(self, prev_velocity, weights, pBest_weights, gBest_weights):
        #weights = self.population[i].layers[j].next_weights
        velocity = []
        for i in range(len(prev_velocity)):
            v1 = self.inertia_C*prev_velocity[i]
            # v1 = 0
            v2 = self.pBest_C*random.uniform(0,1) * (pBest_weights[i] - weights[i])
            v3 = self.gBest_C * random.uniform(0,1) * (gBest_weights[i] - weights[i])

            # if i == 0:
                # print("pBest = weights ", pBest_weights == weights)
                # # print("PBC ", self.pBest_C)
                # print("PBW ", pBest_weights[i])
                # print("W", weights[i])
                # print("v1 ", v1, "\n v2 ", v2, "\n v3 ", v3)

            v = v1 + v2 + v3
            velocity.append(v)
        return velocity

    def run_PSO(self):
        #iteration count
        iteration = 0
        #calculate fitness
        while iteration < 3:
            # print("iteration ", iteration)
            # print("gBest Fitness ", self.gBest_fitness)
            for i in range(len(self.population)):
                fitness = self.population[i].fitness(self.training_data_inputs, self.training_data_outputs)
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
                    # print("gBest updated", self.gBest_fitness)
                    self.gBest = copy.deepcopy(self.population[i])
                    self.gBest_fitness = fitness

            self.update()
            iteration += 1
            print("iteration ", iteration, " fitness ", self.gBest_fitness)

    def run_test(self):
        classifications = []
        for i in range(len(self.test_data_inputs)):
            prediction = self.gBest.predict(self.test_data_inputs[i])
            if len(prediction) > 1:
                index, value = max(enumerate(prediction), key=operator.itemgetter(1))
                index2, value2 = max(enumerate(self.test_data_outputs[i]), key=operator.itemgetter(1))
                classifications.append([index2+1, index+1])

            else:
                classifications.append([self.test_data_outputs[i], prediction[0]])
        return classifications



if __name__ == "__main__":

    #Prepping data ----------------------------
    #abalone
    data_aba = Data_Processing(["abalone",], [8], {})
    data_car = Data_Processing(["car",], [6], {})
    data_img = Data_Processing(["segmentation",], [0], {})
    data_mach = Data_Processing(["machine",], [-1], {})
    data_ff = Data_Processing(["forestfires",], [-1],{})
    data_wine = Data_Processing(["wine",], [-1], {})

    # #PROCESS DATA
    # data_aba.process_data("./data")
    # data_car.process_data("./data")
    # data_img.process_data("./data")
    # data_mach.process_data("./data")
    # data_ff.process_data("./data")
    # data_wine.process_data("./data")

    # #WRITE DATA
    # data_aba.write_data("./processed")
    # data_car.write_data("./processed")
    # data_img.write_data("./processed")
    # data_mach.write_data("./processed")
    # data_ff.write_data("./processed")
    # data_wine.write_data("./processed")

    # data = Data_Processing(["abalone","car","forestfires","machine","segmentation","wine"], [8,6,12,9,0,11], {})
    # data.load_data("./processed")

    # data_aba = data.file_array['abalone']
    # data_car = data.file_array['car']
    # data_img = data.file_array['forestfires']
    # data_mach = data.file_array['machine']
    # data_ff = data.file_array['segmentation']
    # data_wine = data.file_array['wine']

    #LOAD DATA
    data_aba.load_data("./processed")
    data_car.load_data("./processed")
    data_img.load_data("./processed")
    data_mach.load_data("./processed")
    data_ff.load_data("./processed")
    data_wine.load_data("./processed")

    #set test and training data


    #end --------------------------------------

    #POPULATION = 50
    #ITERATION LIMIT = 1000

    #tesing alogorithm ------------------------
    # inputs -> (self, pop_size, mlp_dims, pBest_coeff, gBest_coeff, inertia_coeff, training_data, test_data)


    # data_aba.file_array['abalone'] = data_aba.file_array['abalone']    #slice in to 5
    # data_aba.slicer(5, "abalone")
    # test_data = data_aba.file_array[0]
    # training_data = data_aba.combine(data_aba.file_array[1:])

    # results_file = open("./results/results_pso.txt", "a+")
    # pso10 = particle_swarm(50, [(len(data_aba.file_array[0][0][1:])),29], 2, 2, 0.4, training_data, test_data)
    # pso10.run_PSO()
    # test_results = pso10.run_test()
    # fscore10 = f_score(test_results)
    # print_str= "Abalone 0 hidden layers Fscore: \n" + "Accuracy: " + str(fscore10["Accuracy"]) + ", F1:" + str(fscore10["F1"]) + ", Precision:" + str(fscore10["Precision"]) + ", Recall:" + str(fscore10["Recall"])
    # results_file.write("\n"+print_str)
    # results_file.close()

    # results_file = open("./results/results_pso.txt", "a+")
    # pso11 = particle_swarm(50, [(len(data_aba.file_array[0][0][1:])),30, 29], 2, 2, 0.4, training_data, test_data)
    # pso11.run_PSO()
    # test_results = pso11.run_test()
    # fscore11 = f_score(test_results)
    # print_str= "Abalone 1 hidden layers Fscore: \n" + "Accuracy: " + str(fscore11["Accuracy"]) + ", F1:" + str(fscore11["F1"]) + ", Precision:" + str(fscore11["Precision"]) + ", Recall:" + str(fscore11["Recall"])
    # results_file.write("\n"+print_str)
    # results_file.close()

    # results_file = open("./results/results_pso.txt", "a+")
    # pso12 = particle_swarm(50, [(len(data_aba.file_array[0][0][1:])),30, 30, 29], 2, 2, 0.4, training_data, test_data)
    # pso12.run_PSO()
    # test_results = pso12.run_test()
    # fscore12 = f_score(test_results)
    # print_str= "Abalone 2 hidden layers Fscore: \n" + "Accuracy: " + str(fscore12["Accuracy"]) + ", F1:" + str(fscore12["F1"]) + ", Precision:" + str(fscore12["Precision"]) + ", Recall:" + str(fscore12["Recall"])
    # results_file.write("\n"+print_str)
    # results_file.close()

    # car
    # 4 classes
    # 6 attributes
    print("car")
    data_car.file_array['car'] = data_car.file_array['car']    #slice in to 5
    data_car.slicer(5, "car")
    test_data = data_car.file_array[0]
    training_data = data_car.combine(data_car.file_array[1:])

    results_file = open("./results/results_pso.txt", "a+")
    pso20 = particle_swarm(50, [(len(data_car.file_array[0][0][1:])),4], 2, 2, 0.4, training_data, test_data)
    pso20.run_PSO()
    test_results = pso20.run_test()
    fscore20 = f_score(test_results)
    print_str= "Car 0 hidden layers Fscore: \n" + "Accuracy: " + str(fscore20["Accuracy"]) + ", F1:" + str(fscore20["F1"]) + ", Precision:" + str(fscore20["Precision"]) + ", Recall:" + str(fscore20["Recall"])
    results_file.write("\n"+print_str)
    results_file.close()

    # results_file = open("./results/results_pso.txt", "a+")
    # pso21 = particle_swarm(50, [(len(data_car.file_array[0][0][1:])),10, 4], 2, 2, 0.4, training_data, test_data)
    # pso21.run_PSO()
    # test_results = pso21.run_test()
    # fscore21 = f_score(test_results)
    # print_str= "Car 1 hidden layers Fscore: \n" + "Accuracy: " + str(fscore21["Accuracy"]) + ", F1:" + str(fscore21["F1"]) + ", Precision:" + str(fscore21["Precision"]) + ", Recall:" + str(fscore21["Recall"])
    # results_file.write("\n"+print_str)
    # results_file.close()

    # results_file = open("./results/results_pso.txt", "a+")
    # pso22 = particle_swarm(50, [(len(data_car.file_array[0][0][1:])),10, 10, 4], 2, 2, 0.4, training_data, test_data)
    # pso22.run_PSO()
    # test_results = pso22.run_test()
    # fscore22 = f_score(test_results)
    # print_str= "Car 2 hidden layers Fscore: \n" + "Accuracy: " + str(fscore22["Accuracy"]) + ", F1:" + str(fscore22["F1"]) + ", Precision:" + str(fscore22["Precision"]) + ", Recall:" + str(fscore22["Recall"])
    # results_file.write("\n"+print_str)
    # results_file.close()

    # segmentation
    # 7 classes
    # 19 attributes
    # data_img.file_array['segmentation'] = data_img.file_array['segmentation']    #slice in to 5
    # data_img.slicer(5, "segmentation")
    # test_data = data_img.file_array[0]
    # training_data = data_img.combine(data_img.file_array[1:])

    # results_file = open("./results/results_pso.txt", "a+")
    # pso30 = particle_swarm(50, [(len(data_img.file_array[0][0][1:])),7], 2, 2, 0.4, training_data, test_data)
    # pso30.run_PSO()
    # test_results = pso30.run_test()
    # fscore30 = f_score(test_results)
    # print_str= "segmentation 0 hidden layers Fscore: \n" + "Accuracy: " + str(fscore30["Accuracy"]) + ", F1:" + str(fscore30["F1"]) + ", Precision:" + str(fscore30["Precision"]) + ", Recall:" + str(fscore30["Recall"])
    # results_file.write("\n"+print_str)
    # results_file.close()

    # results_file = open("./results/results_pso.txt", "a+")
    # pso31 = particle_swarm(50, [(len(data_img.file_array[0][0][1:])),14, 7], 2, 2, 0.4, training_data, test_data)
    # pso31.run_PSO()
    # test_results = pso31.run_test()
    # fscore31 = f_score(test_results)
    # print_str= "segmentation 1 hidden layers Fscore: \n" + "Accuracy: " + str(fscore31["Accuracy"]) + ", F1:" + str(fscore31["F1"]) + ", Precision:" + str(fscore31["Precision"]) + ", Recall:" + str(fscore31["Recall"])
    # results_file.write("\n"+print_str)
    # results_file.close()

    # results_file = open("./results/results_pso.txt", "a+")
    # pso32 = particle_swarm(50, [(len(data_img.file_array[0][0][1:])),14, 14, 7], 2, 2, 0.4, training_data, test_data)
    # pso32.run_PSO()
    # test_results = pso32.run_test()
    # fscore32 = f_score(test_results)
    # print_str= "segmentation 2 hidden layers Fscore: \n" + "Accuracy: " + str(fscore32["Accuracy"]) + ", F1:" + str(fscore32["F1"]) + ", Precision:" + str(fscore32["Precision"]) + ", Recall:" + str(fscore32["Recall"])
    # results_file.write("\n"+print_str)
    # results_file.close()


    # # machine
    # # 1 class
    # # 10 attributes
    # data_mach.file_array['machine'] = data_mach.file_array['machine']    #slice in to 5
    # data_mach.slicer(5, "machine")
    # test_data = data_mach.file_array[0]
    # training_data = data_mach.combine(data_mach.file_array[1:])

    # results_file = open("./results/results_pso.txt", "a+")
    # pso40 = particle_swarm(50, [(len(data_mach.file_array[0][0][1:])),1], 2, 2, 0.4, training_data, test_data)
    # pso40.run_PSO()
    # test_results = pso40.run_test()
    # mse40 = mse(test_results)
    # print_str= "machine 0 hidden layers MSE: \n" + str(mse40)
    # results_file.write("\n"+print_str)
    # results_file.close()

    # results_file = open("./results/results_pso.txt", "a+")
    # pso41 = particle_swarm(50, [(len(data_mach.file_array[0][0][1:])),15, 1], 2, 2, 0.4, training_data, test_data)
    # pso41.run_PSO()
    # test_results = pso41.run_test()
    # mse41 = mse(test_results)
    # print_str= "machine 1 hidden layers MSE: \n" + str(mse41)
    # results_file.write("\n"+print_str)
    # results_file.close()

    # results_file = open("./results/results_pso.txt", "a+")
    # pso42 = particle_swarm(50, [(len(data_mach.file_array[0][0][1:])),15, 15, 1], 2, 2, 0.4, training_data, test_data)
    # pso42.run_PSO()
    # test_results = pso42.run_test()
    # mse42 = mse(test_results)
    # print_str= "machine 2 hidden layers MSE: \n" + str(mse42)
    # results_file.write("\n"+print_str)
    # results_file.close()


    # #forestfires
    # # 1 class
    # # 13 attributes
    # # data_ff.file_array['forestfires'] = data_ff.file_array['forestfires']    #slice in to 5
    # # data_ff.slicer(5, "forestfires")
    # # test_data = data_ff.file_array[0]
    # # training_data = data_ff.combine(data_ff.file_array[1:])

    # # results_file = open("./results/results_pso.txt", "a+")
    # # pso50 = particle_swarm(50, [(len(data_ff.file_array[0][0][1:])),1], 2, 2, 0.4, training_data, test_data)
    # # pso50.run_PSO()
    # # test_results = pso50.run_test()
    # # mse50 = mse(test_results)
    # # print_str= "forestfires 0 hidden layers MSE: \n" + str(mse50)
    # # results_file.write("\n"+print_str)
    # # results_file.close()

    # # results_file = open("./results/results_pso.txt", "a+")
    # # pso51 = particle_swarm(50, [(len(data_ff.file_array[0][0][1:])),20, 1], 2, 2, 0.4, training_data, test_data)
    # # pso51.run_PSO()
    # # test_results = pso51.run_test()
    # # mse51 = mse(test_results)
    # # print_str= "forestfires 1 hidden layers MSE: \n" + str(mse51)
    # # results_file.write("\n"+print_str)
    # # results_file.close()

    # # results_file = open("./results/results_pso.txt", "a+")
    # # pso52 = particle_swarm(50, [(len(data_ff.file_array[0][0][1:])),20, 20, 1], 2, 2, 0.4, training_data, test_data)
    # # pso52.run_PSO()
    # # test_results = pso52.run_test()
    # # mse52 = mse(test_results)
    # # print_str= "forestfires 2 hidden layers MSE: \n" + str(mse52)
    # # results_file.write("\n"+print_str)
    # # results_file.close()

    #wine
    # 1 class
    # 12 attributes
    print("\n wine")
    data_wine.file_array['wine'] = data_wine.file_array['wine']    #slice in to 5
    data_wine.slicer(5, "wine")
    test_data = data_wine.file_array[0]
    training_data = data_ff.combine(data_wine.file_array[1:])

    results_file = open("./results/results_pso.txt", "a+")
    pso60 = particle_swarm(50, [(len(data_wine.file_array[0][0][1:])),1], 2, 2, 0.4, training_data, test_data)
    pso60.run_PSO()
    test_results = pso60.run_test()
    mse60 = mse(test_results)
    print_str= "wine 0 hidden layers MSE: \n" + str(mse60)
    results_file.write("\n"+print_str)
    results_file.close()

    # results_file = open("./results/results_pso.txt", "a+")
    # pso61 = particle_swarm(50, [(len(data_wine.file_array[0][0][1:])),18, 1], 2, 2, 0.4, training_data, test_data)
    # pso61.run_PSO()
    # test_results = pso61.run_test()
    # mse61 = mse(test_results)
    # print_str= "wine 1 hidden layers MSE: \n" + str(mse61)
    # results_file.write("\n"+print_str)
    # results_file.close()

    # results_file = open("./results/results_pso.txt", "a+")
    # pso62 = particle_swarm(50, [(len(data_wine.file_array[0][0][1:])),18, 18, 1], 2, 2, 0.4, training_data, test_data)
    # pso62.run_PSO()
    # test_results = pso62.run_test()
    # mse62 = mse(test_results)
    # print_str= "wine 2 hidden layers MSE: \n" + str(mse62)
    # results_file.write("\n"+print_str)
    # results_file.close()


    #end --------------------------------------
