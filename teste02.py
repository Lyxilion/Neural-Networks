import numpy as np
import random as rdm
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# derivative of sigmoid
# sigmoid(y) * (1.0 - sigmoid(y))
# the way we use this y is already sigmoided
def dsigmoid(y):
    return sigmoid(y) * (1.0 - sigmoid(y))

X = [0.1,0.51]
Y = [0]

axe = [[rdm.randint(1,10)/10,rdm.randint(1,10)/10,rdm.randint(1,10)/10],[rdm.randint(1,10)/10,rdm.randint(1,10)/10,rdm.randint(1,10)/10]], [rdm.randint(1,10)/10,rdm.randint(1,10)/10,rdm.randint(1,10)/10]
error = -1
Ycalc = [0,0]
dweight = [0,0,0]
dHsum = [0,0,0]
dweight2 = [[0,0,0],[0,0,0]]
neu = [[0,0],[0,0],[0,0]]   #[sum][sigmoid(sum)]

for i in range(100):
    for i in range(len(neu)):
        neu[i][0] = X[0] * axe[0][0][i] + X[1] * axe[0][1][i]
        neu[i][1] = sigmoid(neu[i][0])
    Ycalc[0] = neu[0][1] * axe[1][0] + neu[1][1] * axe[1][1] + neu[2][1] * axe[1][2]
    Ycalc[1] = sigmoid(Ycalc[0])


    error = Y[0] - Ycalc[1]
    deltaOutS = Ycalc[1] * error
    for i in range(len(dweight)) :
        dweight[i] = deltaOutS / neu[i][1]
        axe[1][i] = axe[1][i] + dweight[i]
        dHsum[i] = (deltaOutS / (axe[1][i] - dweight[i])) * dsigmoid(neu[i][0])
    for i in range(len(dweight2)) :
        for j in range(len(dweight2[i])) :
            dweight2[i][j] = dHsum[j] / X[i]
            axe[0][i][j] += dweight2[i][j]

    for i in range(len(neu)) :
        neu[i][0] = X[0] * axe[0][0][i] + X[1] * axe[0][1][i]
        neu[i][1] = sigmoid(neu[i][0])


    Ycalc[0] = neu[0][1] * axe[1][0] + neu[1][1] * axe[1][1] + neu[2][1] * axe[1][2]
    Ycalc[1] = sigmoid(Ycalc[0])

print("WTFWTFWTF",Ycalc)
print(axe)

X = [0.5,0.3]
for i in range(len(neu)):
    neu[i][0] = X[0] * axe[0][0][i] + X[1] * axe[0][1][i]
    neu[i][1] = sigmoid(neu[i][0])

Ycalc[0] = neu[0][1] * axe[1][0] + neu[1][1] * axe[1][1] + neu[2][1] * axe[1][2]
Ycalc[1] = sigmoid(Ycalc[0])

print(Ycalc)
print(dweight2)