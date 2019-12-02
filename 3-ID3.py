#import libraries
import numpy as np
import pandas as pd
import math

#constructor for each node
class Node:
    def __init__(self, l):
        self.label = l
        self.branches = {}

#calculate and return entropy
def calc_entropy(data):
    n = len(data)
    pos = len(data.loc[data["Play Tennis"]=='Y'])
    neg = len(data.loc[data["Play Tennis"]=='N'])
    entropy = 0

    if pos>0:
        entropy = (-1)*(pos/float(n))*(math.log(pos,2)-(math.log(n,2)))
    if neg>0:
        entropy += (-1)*(neg/float(n))*(math.log(neg,2)-(math.log(n,2)))
    
    return entropy

#gain = entropy(parent_data) - ((#subset/#set)*entropy(subset) for all values of attr)
def gain(s, data, attr):
    values = set(data[attr])
    gain = s
    for val in values:
        gain -= len(data.loc[data[attr]==val])/float(len(data)) * calc_entropy(data.loc[data[attr]==val])
    
    return gain

#get the split attribute based on maximum gain
def get_attr(data):
    entropy_s = calc_entropy(data)
    attr = ""
    max_gain = 0

    #calculate gain for each attribute return the attribute with maximum gain
    for col in data.columns[:-1]:
        g = gain(entropy_s, data, col)

        if g>max_gain:
            max_gain = g
            attr = col

    return attr

#tree creation
def id3(data):
    root = Node("NULL")
    #Pure class
    if(calc_entropy(data)==0):
        if(len(data.loc[data["Play Tennis"]=='Y']) == len(data)):
            root.label = 'Y'
            return root
        else:
            root.label = 'N'
            return root

    split_attr = get_attr(data)
    root.label = split_attr
    #possible values of the attribute
    values = set(data[split_attr])

    for val in values:
        root.branches[val] = id3(data.loc[data[split_attr]==val].drop(split_attr, axis=1))
    return root


def main():
    data = pd.read_csv('datasets/3.csv')
    tree = id3(data)

    #print(tree.branches['Sunny'].branches['Normal'].label)

main()