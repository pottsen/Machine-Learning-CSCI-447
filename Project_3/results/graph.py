#import matplotlib
import re


file = open("./results.txt", "r")

str = ""
for i in file:
    str+= i

arr = str.split("\n\n")
dicts = []
for i in arr:
    network = i[:3]
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
    for j in dict_list:
        if (i['network']==j['network'] and i['dataset']==j['dataset'] and i['Hidden-layers']==j['Hidden-layers']):
            in_list = True
            try:
                i['MSE']+=j['MSE']
                i['total-folds']+=1
            except:
                i['F-score']+=j['F-score']
                i['total-folds']+=1
    if(not in_list):
        dict_list.append(i)
for i in dict_list:
    try:
        i['MSE']/=i['total-folds']
    except:
        i['F-score']/=i['total-folds']
    del i['fold']

print(dict_list)
