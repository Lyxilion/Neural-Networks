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
Ycalc = [0,0]
dweight2 = []

#adjusting the size of lists
for i in range(len(X)) :
    Liste = []
    for j in range(len(X)+1) :
        Liste.append(rdm.randint(1,10)/10)
    axe[0].append(Liste)
    axe[1].append(rdm.randint(1,10)/10)
    neu.append([None,None])
for i in range(len(axe[1])) :
    dweight.append(None)
    dHsum.append(None)
for i in range(len(X)):
    Liste = []
    for j in range(len(axe[0])) :
        Liste.append(None)
    dweight2.append(Liste)


#LEARNING PHASE
for i in range(100):
    #Begin of foward progression
    for i in range(len(neu)): #calculing the neuronnes
        neu[i][0] = X[0] * axe[0][0][i] + X[1] * axe[0][1][i]
        neu[i][1] = sigmoid(neu[i][0])

    Ycalc[0] = neu[0][1] * axe[1][0] + neu[1][1] * axe[1][1] + neu[2][1] * axe[1][2]
    Ycalc[1] = sigmoid(Ycalc[0])
    #End of foward progresion


    #Beining of backward progresion
    error = Y[0] - Ycalc[1]
    deltaOutS = Ycalc[1] * error

    for i in range(len(dweight)) :#adjust axe[1]
        dweight[i] = deltaOutS / neu[i][1]
        axe[1][i] = axe[1][i] + dweight[i]
        dHsum[i] = (deltaOutS / (axe[1][i] - dweight[i])) * dsigmoid(neu[i][0])

        """Dhsum pas homogene a ce qu'il faut """



    for i in range(len(dweight2)) : #Ajust axe[0]
        for j in range(len(dweight2[i])) :
            dweight2[i][j] = dHsum[j] / X[i]
            axe[0][i][j] += dweight2[i][j]

    for i in range(len(neu)): #calculation new neurones
        neu[i][0] = X[0] * axe[0][0][i] + X[1] * axe[0][1][i]
        neu[i][1] = sigmoid(neu[i][0])
    #End of backward progression

Ycalc[0] = neu[0][1] * axe[1][0] + neu[1][1] * axe[1][1] + neu[2][1] * axe[1][2]
Ycalc[1] = sigmoid(Ycalc[0])

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
