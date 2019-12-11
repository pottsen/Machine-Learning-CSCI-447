from mlp import MLP
from data_processing import Data_Processing
from evaluations import  f_score, mse
import random


def main():
    #define data set names
    sets = ["abalone","car","segmentation","forestfires","machine","wine"]
    #define mlp dimensions for each data set
    input = [8,6,19,12,9,11]
    hidden_layer = [30,10,20,14,15,18]
    outputs = [30,4,7,1,1,1]

    encode_lay = {0:[7,5,18,11,8,10],1:[5,4,13,8,6,8],2:[4,3,9,6,5,6]}

    file = open("./results/results.txt", "w+")
    dict = {}

    for i in range(len(sets)):
        dict.update({sets[i]:{"Baseline":{},"1Layer":{},"2Layer":{},"3Layer":{}}})

    for set in range(len(sets)):
        #for encode_layers in range(3):
        fold = 3
        #for fold in range(5):
        data = Data_Processing([sets[set]], [], {})
        data.load_data("./processed")

        #randomize data order
        data.slicer(5, sets[set])
        data = data.combine(data.file_array)
        random.shuffle(data)
        random.shuffle(data)
        random.shuffle(data)

        #pull off classes
        targets = []
        for point in range(len(data)):
            targets.append(data[point][0])
            data[point] = data[point][1:]

        if(set<3):
            for point in range(len(targets)):
                cla = targets[point]
                targets[point] = [0]*outputs[set]
                targets[point][int(cla)] = 1

        #slice into training data (4/5) and test data (1/5) for each 'fold' (1-5) these sets will be different
        b1 = int(len(data)*fold/5)
        b2 = int(len(data)*(fold+1)/5)
        test_data = data[b1 : b2]
        training_data = data[:b1]
        training_data += data[b2:]
        test_targets = targets[b1 : b2]
        training_targets = targets[:b1]
        training_targets += targets[b2:]

        print("Raw: "+str(test_data[0])+ " Class: "+str(test_targets[0]))

        classifier = [input[set], hidden_layer[set], outputs[set]]
        classifier = MLP(classifier)
        classifier.backprop(training_data,training_targets,5000,0.5,10)

        guesses = []
        if(outputs[set] == 1):
            for i in range(len(test_data)):
                guess = classifier.predict(test_data[i])[0]
                guesses.append([targets[i],guess])
            score = {"MSE": mse(guesses)}
        else:
            for i in range(len(test_data)):
                guess = 0
                guessidx = 0
                actual = 0
                actualidx = 0
                predict = classifier.predict(test_data[i])
                for j in range(len(predict)):
                    if(predict[j]>guess):
                        guess = predict[j]
                        guessidx = j
                    if(test_targets[i][j]>actual):
                        actual = test_targets[i][j]
                        actualidx = j
                guesses.append([actualidx,guessidx])
            score = {"F1": f_score(guesses)}
        dict[sets[set]]["Baseline"] = score
        print(score)


        for encode_layers in range(3):
            if(encode_layers==0):
                encoder = [input[set], encode_lay[0][set], input[set]]
                encoder = MLP(encoder)
            if(encode_layers==1):
                encoder = [input[set], encode_lay[0][set], encode_lay[1][set], encode_lay[0][set], input[set]]
                encoder = MLP(encoder)
            if(encode_layers==2):
                encoder = [input[set],  encode_lay[0][set],  encode_lay[1][set], encode_lay[2][set], encode_lay[1][set], encode_lay[0][set], input[set]]
                encoder = MLP(encoder)

            encoder.backprop(training_data,training_data,10000,0.5,100)
            reconstructions = []
            for i in range(len(test_data)):
                reconstructions.append(encoder.predict(test_data[i], regression = True))

            print("Reconstructed: "+str(reconstructions[0]))

            guesses1 = []
            if(outputs[set] == 1):
                for i in range(len(test_data)):
                    guess = classifier.predict(reconstructions[i])[0]
                    guesses1.append([targets[i],guess])
                score1 = {"MSE": mse(guesses1)}
            else:
                for i in range(len(test_data)):
                    guess = 0
                    guessidx = 0
                    actual = 0
                    actualidx = 0
                    predict = classifier.predict(reconstructions[i])
                    for j in range(len(predict)):
                        if(predict[j]>guess):
                            guess = predict[j]
                            guessidx = j
                        if(test_targets[i][j]>actual):
                            actual = test_targets[i][j]
                            actualidx = j
                    guesses1.append([actualidx,guessidx])
                score1 = {"F1": f_score(guesses1)}
            print("Guess: "+str(guesses1[0][1]))
            dict[sets[set]][str(encode_layers+1)+"Layer"] = score1
            print(score1)

    file.write(str(dict))

    # training = [[0,0,0],[0,1,1],[1,0,1],[1,1,0]]
    # training2 = [[0,0,0],[0,1,1],[1,0,1],[1,1,0],[1,0,0],[0,0,1]]
    # outputs2 = [[1,1,1],[1,0,0],[0,1,0],[0,0,1],[0,1,1],[1,1,0]]
    #
    # mlp = MLP([3,4,3])
    # guess = mlp.predict(training2[0], backprop = True)
    # print(mlp.predict(training2[0]))
    # mlp.backprop(training2,outputs2,0.5)
    #
    #
    # print(mlp.predict(training2[0]))
    # print(mlp.predict(training2[1]))
    # print(mlp.predict(training2[2]))

    #----------------------------

    # training2 = [[0,0],[0,1],[1,0],[1,1]]
    # training2 = [[0,0],[0,1],[1,0],[1,1]]
    # outputs2 = [[0],[1],[1],[0]]
    #
    # mlp = MLP([2,4,1])
    # guess = mlp.predict(training2[0], backprop = True)
    # print(mlp.predict(training2[0]))
    # mlp.backprop(training2,outputs2,0.5)
    #
    #
    # print(mlp.predict(training2[0]))
    # print(mlp.predict(training2[1]))
    # print(mlp.predict(training2[2]))
    # print(mlp.predict(training2[3]))
def main2():
    pass
main()
