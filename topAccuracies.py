from json import loads
from pandas import DataFrame as df
import pandas as pd
import matplotlib
from os import mkdir
from os.path import isdir
from os import listdir
from sys import argv

# Gather results from given directory
inputDir = argv[1]
allFiles = listdir(inputDir)
allFiles = [allFiles[i] for i in range(len(allFiles)) if "result-" in allFiles[i]]
allDicts = []
print("Gathering contents of {} files...".format(len(allFiles)))
for j in allFiles:
    with open("{}/{}".format(inputDir,j)) as infile:
        contents = infile.read().strip()
            
    # contents = contents.replace("'", '"')
    if (len(contents) == 0):
        continue
    # print(j)
    # print(contents)
    try:
        contents = list(map(
        lambda row: loads(row.replace("'", '"').replace("True", "true").replace("False", "false")), contents.split("\n")))
    except Exception as e:
        print(e)
        print(contents)
    allDicts.extend(contents)

allData = df(allDicts)
allData['acc'] = allData['acc'].astype(float)
# Find the best accuracy
# Top 5
bestOf = int(argv[2])
if bestOf > len(allDicts):
    bestOf = len(allDicts)
if len(argv) == 4:
    worstOf = int(argv[3])
    if worstOf > len(allDicts):
        worstOf = len(allDicts)
else:
    worstOf = 0
print("Top five networks and their configurations")
# print(allData)
# Extract elements from lists
def extract1(row):
    return row.map(extract2)
def extract2(val):
    if isinstance(val, list) and len(val) == 1:
        return val[0]
    return val
# Get the bestOf largest accuracies
res = allData.nlargest(bestOf, columns='acc')#.drop('tacc', axis=1)
res.sort_values(by='acc')
res.index = range(1, res.shape[0]+1)
v = res.columns.to_list()
del v[v.index('acc')]
v.insert(0, 'acc')
res = res[v]
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
res  = res.apply(extract1)
print(res.drop(['gpu', 'tacc', 'time', 'data_multiplier'], axis=1))
if worstOf > 0:
    perc = .12
    print("Worst {} above {:2.0f}%".format(worstOf, perc*100))
    res2 = allData.loc[allData['acc'] > 0.12]
    res2 = res2.nsmallest(worstOf, columns='acc')
    res2.index = range(1, res2.shape[0]+1)
    res2 = res2[v]
    res2 = res2.apply(extract1)
    print(res2.drop(['gpu', 'tacc', 'time', 'data_multiplier'], axis=1))

# # Make plots of the top X
import matplotlib.pyplot as plt

try:
    mkdir("figures/")
except:
    pass
for i, d in res.iterrows():
    fig, ax = plt.subplots()
    ax.plot(d['tacc'])
    ax.set_title("testing accuracy = {:4.3}, nlayers = {}, dropout = {}\nbatch size = {}, learning rate = {}, neurons = {}".format(
        d['acc'], d['num_layers'],d['dropout'],d['batch_size'], d["learning_rate"], 
        d["neurons"]))
    ax.set_xlabel('epochs')
    ax.set_ylabel('training accuracy')
    fig.savefig("figures/{}.png".format(d['hash']))
    plt.close()
