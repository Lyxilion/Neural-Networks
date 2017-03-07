import numpy as np
import random as rdm
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def dsigmoid(y):
    return sigmoid(y) * (1.0 - sigmoid(y))

X = [1,1]   #Input
Y = [0.5]     #Output

axe1 = [[rdm.randint(0,10)/10,rdm.randint(0,10)/10,rdm.randint(0,10)/10],[rdm.randint(0,10)/10,rdm.randint(0,10)/10,rdm.randint(0,10)/10]]
axe2 = [rdm.randint(0,10)/10,rdm.randint(0,10)/10,rdm.randint(1,10)/10]

neu = [[0,0],[0,0],[0,0]]   #[sum][sigmoid(sum)]

neu[0][0] = X[0] * axe1[0][0] + X[1] * axe1[1][0]
neu[1][0] = X[0] * axe1[0][1] + X[1] * axe1[1][1]
neu[2][0] = X[0] * axe1[0][2] + X[1] * axe1[1][2]

neu[0][1] = sigmoid(neu[0][0])
neu[1][1] = sigmoid(neu[1][0])
neu[2][1] = sigmoid(neu[2][0])

Ycalc = [0,0]
Ycalc[0] = neu[0][1] * axe2[0] + neu[1][1] * axe2[1] + neu[2][1] * axe2[2]
Ycalc[1] = sigmoid(Ycalc[0])

print("Ycalc = ",Ycalc[0])
print("S(Ycalc) = ",Ycalc[1])

error = Y[0] - Ycalc[1]
deltaOutS = Ycalc[1] * error
dweight = [0,0,0]


dweight[0] = deltaOutS / neu[0][1]
dweight[1] = deltaOutS / neu[1][1]
dweight[2] = deltaOutS / neu[2][1]

axe2[0] = axe2[0] + dweight[0]
axe2[1] = axe2[1] + dweight[1]
axe2[2] = axe2[2] + dweight[2]

print('error = ',error)
print("deltaOutS = ",deltaOutS)
print("dweight = ", dweight)

print("axe2 fix = ",axe2)

dHsum = [0,0,0]

dHsum[0] = (deltaOutS / (axe2[0] - dweight[0])) * dsigmoid(neu[0][0])
dHsum[1] = (deltaOutS / (axe2[1] - dweight[1])) * dsigmoid(neu[1][0])
dHsum[2] = (deltaOutS / (axe2[2] - dweight[2])) * dsigmoid(neu[2][0])
print(neu)
print(dHsum)

dweight2 = [[0,0,0],[0,0,0]]

dweight2[0][0] = dHsum[0] / X[0]
dweight2[0][1] = dHsum[1] / X[0]
dweight2[0][2] = dHsum[2] / X[0]
dweight2[1][0] = dHsum[0] / X[1]
dweight2[1][1] = dHsum[1] / X[1]
dweight2[1][2] = dHsum[2] / X[1]


axe1[0][1] += dweight2[0][1]
axe1[0][2] += dweight2[0][2]
axe1[1][0] += dweight2[1][0]
axe1[1][1] += dweight2[1][2]
axe1[1][2] += dweight2[1][2]

print(axe1)


neu[0][0] = X[0] * axe1[0][0] + X[1] * axe1[1][0]
neu[1][0] = X[0] * axe1[0][1] + X[1] * axe1[1][1]
neu[2][0] = X[0] * axe1[0][2] + X[1] * axe1[1][2]

neu[0][1] = sigmoid(neu[0][0])
neu[1][1] = sigmoid(neu[1][0])
neu[2][1] = sigmoid(neu[2][0])

Ycalc[0] = neu[0][1] * axe2[0] + neu[1][1] * axe2[1] + neu[2][1] * axe2[2]
Ycalc[1] = sigmoid(Ycalc[0])

print("Ycalc = ",Ycalc[0])
print("S(Ycalc) = ",Ycalc[1])

error = Y[0] - Ycalc[1]
deltaOutS = Ycalc[1] * error
dweight = [0,0,0]

dweight[0] = deltaOutS / neu[0][1]
dweight[1] = deltaOutS / neu[1][1]
dweight[2] = deltaOutS / neu[2][1]

axe2[0] = axe2[0] + dweight[0]
axe2[1] = axe2[1] + dweight[1]
axe2[2] = axe2[2] + dweight[2]

print('error = ',error)
print("deltaOutS = ",deltaOutS)
print("dweight = ", dweight)

print("axe2 fix = ",axe2)

dHsum = [0,0,0]

dHsum[0] = (deltaOutS / (axe2[0] - dweight[0])) * dsigmoid(neu[0][0])
dHsum[1] = (deltaOutS / (axe2[1] - dweight[1])) * dsigmoid(neu[1][0])
dHsum[2] = (deltaOutS / (axe2[2] - dweight[2])) * dsigmoid(neu[2][0])
print(neu)
print(dHsum)

dweight2 = [[0,0,0],[0,0,0]]

dweight2[0][0] = dHsum[0] / X[0]
dweight2[0][1] = dHsum[1] / X[0]
dweight2[0][2] = dHsum[2] / X[0]
dweight2[1][0] = dHsum[0] / X[1]
dweight2[1][1] = dHsum[1] / X[1]
dweight2[1][2] = dHsum[2] / X[1]


axe1[0][0] += dweight2[0][0]
axe1[0][1] += dweight2[0][1]
axe1[0][2] += dweight2[0][2]
axe1[1][0] += dweight2[1][0]
axe1[1][1] += dweight2[1][1]
axe1[1][2] += dweight2[1][2]

print(axe1)


neu[0][0] = X[0] * axe1[0][0] + X[1] * axe1[1][0]
neu[1][0] = X[0] * axe1[0][1] + X[1] * axe1[1][1]
neu[2][0] = X[0] * axe1[0][2] + X[1] * axe1[1][2]

neu[0][1] = sigmoid(neu[0][0])
neu[1][1] = sigmoid(neu[1][0])
neu[2][1] = sigmoid(neu[2][0])

Ycalc[0] = neu[0][1] * axe2[0] + neu[1][1] * axe2[1] + neu[2][1] * axe2[2]
Ycalc[1] = sigmoid(Ycalc[0])

print("Ycalc = ",Ycalc[0])
print("S(Ycalc) = ",Ycalc[1])
