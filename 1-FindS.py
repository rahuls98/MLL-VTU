import csv
import numpy as np
import pandas as pd
hypo = ['%','%','%','%','%','%']

lines = csv.reader(open('datasets/1_2.csv'))
data = list(lines)
print("\nTraining data:\n")
for row in data:
    print(str(row))

data.pop(0)
hypo = data[0][:-1]

for i in range(1, len(data)):
    if data[i][-1]=='N':
        continue
    
    for j,val in enumerate(data[i][:-1]):
        if hypo[j] != val:
            hypo[j] = '?'
            continue
        continue

print("\nMost specific hypothesis for given data: ", hypo)

i = input("\nEnter test instance: ")
test = i.split(',')
flag = True
for i in range(len(hypo)):
    if hypo[i]=='?':
        continue
    elif test[i] == hypo[i]:
        continue
    else:
        flag = False
        break

print('Y') if(flag) else print('N')
