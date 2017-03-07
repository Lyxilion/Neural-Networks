import numpy as np
import random as rdm
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def dsigmoid(y):
    return sigmoid(y) * (1.0 - sigmoid(y))

X = [5.1,3.5,1.4,0.2]
Y = [1,1,0]
#Creation of var
axe = [],[]
neu = []
dweight = []
dHsum = []
Ycalc = [0,0],[0,0],[0,0]
dweight2 = []
error = [None,None,None]
deltaOutS = [None,None,None]

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

#LEARNING PHASE
for i in range(10):
    #Begin of foward progression
    for i in range(len(neu)): #calculing the neuronnes
        neu[i][0] = X[0] * axe[0][0][i] + X[1] * axe[0][1][i]
        neu[i][1] = sigmoid(neu[i][0])


    for i in range(len(Ycalc)) :
        for j in range(len(neu)):
            Ycalc[i][0] += neu[j][1] * axe[1][j][i]
            Ycalc[i][1] = sigmoid(Ycalc[i][0])

    '''print(Ycalc)'''
    #End of foward progresion

    #Beining of backward progresion
    for i in range(len(error)):
        error[i] = Y[i] - Ycalc[i][1]
        deltaOutS[i] = Ycalc[i][1] * error[i]

    for i in range(len(dweight)) :#adjust axe[1]
        for j in range(len(dweight[i])):
            dweight[i][j] = deltaOutS[i] / neu[j][1]
            axe[1][j][i] = axe[1][j][i] + dweight[i][j]
            dHsum[i][j] = (deltaOutS[0] / (axe[1][j][i] - dweight[i][j])) * dsigmoid(neu[i][0])

    for i in range(len(dweight2)) : #Ajust axe[0]
        for j in range(len(dweight2[i])) :
            dweight2[i][j] = dHsum[j][i] / X[i]
            axe[0][j][i] += dweight2[i][j]
    print(dHsum)

    print("dweight2 : ", dweight2)
    for i in range(len(neu)): #calculation new neurones
        neu[i][0] = X[0] * axe[0][0][i] + X[1] * axe[0][1][i]
        neu[i][1] = sigmoid(neu[i][0])
    #End of backward progression

Ycalc[0][0] = neu[0][1] * axe[1][0] + neu[1][1] * axe[1][1] + neu[2][1] * axe[1][2]
Ycalc[0][1] = sigmoid(Ycalc[0][0])

print("axe :", axe)
print("Ycalc :",Ycalc)


#Testing phase
X = [6.7,3.3,5.7,2.5]
for i in range(len(neu)):
    neu[i][0] = X[0] * axe[0][0][i] + X[1] * axe[0][1][i]
    neu[i][1] = sigmoid(neu[i][0])
Ycalc[0] = neu[0][1] * axe[1][0] + neu[1][1] * axe[1][1] + neu[2][1] * axe[1][2]
Ycalc[1] = sigmoid(Ycalc[0])

print("Test : ",Ycalc)