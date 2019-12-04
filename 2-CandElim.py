#import libraries
import numpy as np
import pandas as pd

#read data
dataset = pd.read_csv('datasets/1_2.csv')

#separate data and targets
data = np.array(dataset.iloc[:,:-1])
targets = np.array(dataset.iloc[:,-1])

#initialize S and G
s = data[0]
g = []

#create a len(s)xlen(s) nd array
for i in range(len(s)):
    l = []
    for j in range(len(s)):
        l.append('?')

    g.append(l)

#go through all concepts one by one
for ind,row in enumerate(data):
    #if target of concept is 'P'
    if targets[ind] == 'P':
        for j in range(len(s)):
            if row[j] != s[j]:
                s[j] = '?'
            if s[j]!=g[j][j] and g[j][j]!='?'
                g[j][j] = '?'

    #if target of concept is 'N'
    if targets[ind] == 'N':
        for j in range(len(s)):
            if row[j] != s[j]:
                g[j][j] = s[j]

popindex = []
final_g = []
for i in range(len(s)):
    if g[i] != ['?']*len(s):
        final_g.append(g[i])

print(s)
print(final_g)
