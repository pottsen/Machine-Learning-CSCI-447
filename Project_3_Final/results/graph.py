import re
import matplotlib.pyplot as plt
import numpy as np

def get_data_RBN():

    file = open("./RBN.txt", "r")

    str = ""
    for i in file:
        str+= i

    arr = str.split("\n\n")
    #print(arr)
    dicts = []
    for i in arr:
        #print(i)
        network = "RBN"
        dataset = re.search(r"(?:for )(\w*)(?: fold)", i).group(1)
        fold = re.search(r"(?:fold: *)(\d*)", i).group(1)
        if('edited' in i):
            algo = 'edited'
        elif('kmeans' in i):
            algo = 'kmeans'
        elif('kmedoids' in i):
            algo = 'kmedoids'
        try:
            mse = re.search(r"(?:MSE: \[)([\d\.]*)", i).group(1)
            fold = {'network':network,'dataset':dataset,'fold':fold,'MSE':float(mse), 'total-folds':1, 'algo':algo}
        except:
            f = re.search(r"(?:F-score: )([\d\.]*)", i).group(1)
            fold = {'network':network,'dataset':dataset,'fold':fold, 'F-score':float(f), 'total-folds':1, 'algo':algo}
        dicts.append(fold)
    return dicts





def get_data_MLP():
    file = open("MLP.txt", "r")

    str = ""
    for i in file:
        str+= i

    arr = str.split("\n\n")
    dicts = []
    for i in arr:
        network = "MLP"
        dataset = re.search(r"(?:for )(\w*)(?: fold)", i).group(1)
        fold = re.search(r"(?:fold: *)(\d*)", i).group(1)
        dims = re.search(r"(?:\[)(.*)(?:\])", i).group(1)
        try:
            mse = re.search(r"(?:MSE: )([\d\.]*)", i).group(1)
            fold = {'network':network,'dataset':dataset,'fold':fold,'Hidden-layers':dims, 'MSE':float(mse), 'total-folds':1}
        except:
            f = re.search(r"(?:F-score: )([\d\.]*)", i).group(1)
            fold = {'network':network,'dataset':dataset,'fold':fold,'Hidden-layers':dims, 'F-score':float(f), 'total-folds':1}
        dicts.append(fold)

    dict_list = []
    for i in dicts:
        in_list = False
        for j in range(len(dict_list)):
            if (i['dataset']==dict_list[j]['dataset'] and i['Hidden-layers']==dict_list[j]['Hidden-layers']):
                in_list = True
                try:
                    dict_list[j]['MSE']+=i['MSE']
                    dict_list[j]['total-folds']+=1
                except:
                    dict_list[j]['F-score']+=i['F-score']
                    dict_list[j]['total-folds']+=1
        if(not in_list):
            dict_list.append(i)


    for i in range(len(dict_list)):
        try:
            dict_list[i]['MSE']/=dict_list[i]['total-folds']
        except:
            dict_list[i]['F-score']/=dict_list[i]['total-folds']
        del dict_list[i]['fold']
    return dict_list




if __name__ == "__main__":
    dataMLP = get_data_MLP()
    dataMLP += get_data_RBN()
    print(dataMLP)
    graph1=[[],[],[],[]]  #MLP F
    graph2=[[],[],[],[]]  #MLP MSE
    graph3=[[],[],[],[]]  #RBN F
    graph4=[[],[],[],[]]  #RBN MSE

    graphs = [graph1,graph2,graph3,graph4]

    for i in dataMLP:
        if(i['network']=='MLP'):
            try:
                if ((len(i["Hidden-layers"]))== 0):
                    graph1[0].append(i["MSE"])
                elif len(i["Hidden-layers"]) != 0 and not "," in i["Hidden-layers"]:
                    graph1[1].append(i["MSE"])
                else:
                    graph1[2].append(i["MSE"])
                graph1[3] =["machine", "forestfires", "wine"]
            except:
                if len(i["Hidden-layers"]) == 0:
                    graph2[0].append(i["F-score"])
                elif len(i["Hidden-layers"])!= 0 and not "," in i["Hidden-layers"]:
                    graph2[1].append(i["F-score"])
                else:
                    graph2[2].append(i["F-score"])
                graph2[3] = ["abalone", "car", "segmentation"]
        elif(i['network']=='RBN'):
            try:
                if i["algo"] == 'edited':
                    graph3[0].append(i["MSE"])
                elif i["algo"] == 'kmeans':
                    graph3[1].append(i["MSE"])
                elif i["algo"] == 'kmedoids':
                    graph3[2].append(i["MSE"])
                graph3[3] = ["machine", "forestfires", "wine"]
            except:
                if i["algo"] == 'edited':
                    graph4[0].append(i["F-score"])
                elif i["algo"] == 'kmeans':
                    graph4[1].append(i["F-score"])
                elif i["algo"] == 'kmedoids':
                    graph4[2].append(i["F-score"])
                graph4[3] = ["abalone", "car", "segmentation"]

    print(graph1)
    print(graph2)
    print(graph3)
    print(graph4)

    titles = ['MLP Classification','MLP Regression','RBF Regression','RBF Classification']
    error = ['MSE', 'F-score','MSE','F-score']
    inc = 0
    for i in graphs:
        # width of the bars
        barWidth = 0.3

        # The x position of bars
        r1 = np.arange(len(i[2]))
        r2 = [x + barWidth for x in r1]
        r3 = [y + 2*barWidth for y in r1]

        plt.figure(figsize=(10,6))
        # Create blue bars
        r = plt.bar(r1, i[0], width = barWidth, color = 'blue', edgecolor = 'black', capsize=7, label='poacee')

        # Create cyan bars
        r+=plt.bar(r2, i[1], width = barWidth, color = 'cyan', edgecolor = 'black', capsize=7, label='sorgho')

        # Create green bars
        r+=plt.bar(r3, i[2], width = barWidth, color = 'green', edgecolor = 'black', capsize=7, label='null')

        # general layout
        plt.xticks([r + barWidth for r in range(len(i[2]))], i[3])
        plt.ylabel(error[inc])
        plt.title(titles[inc])
        #plt.legend()

        for h in r:
            height = h.get_height()
            plt.text(h.get_x(), 1.01*height, '{0:.2f}'.format(height), va='bottom')
        # Show graphic

        plt.show()

        inc +=1
