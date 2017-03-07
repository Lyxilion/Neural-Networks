import numpy as np
import random as rdm
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def dsigmoid(y):
    return sigmoid(y) * (1.0 - sigmoid(y))

X = [5.1,3.5,1.4,0.2]
Y = [1]
#Creation of var
axe = [],[]
neu = []
dweight = []
dHsum = []
Ycalc = [0,0],[0,0],[0,0]
dweight2 = []

#adjusting the size of lists
SIZE = len(X)
for i in range(SIZE) :
    List = []
    List2 = []
    for j in range(SIZE) :
        List.append(rdm.randint(1,10)/10)
    for k in range(SIZE-1) :
        List2.append(rdm.randint(1,10)/10)
    axe[0].append(List)
    axe[1].append(List2)
    neu.append([None,None])
for i in range(len(Ycalc)) :
    List =[]
    for i in range(len(axe[1])):
        List.append(None)
    dweight.append(List)
    dHsum.append(List)
for i in range(SIZE):
    List = []
    for j in range(len(axe[0])-1) :
        List.append(None)
    dweight2.append(List)


print("dweight :",dweight2)
print("dHsum :",dHsum)
print(dweight2)