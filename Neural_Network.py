"""
Made by Lyxilion
"""

#IMPORT
import numpy as np
import random as rdm
import os as os

#FUNCTIONS
def read(path = 'C:\\Users\Lyxilion\Documents\\Neurone\\Neural-Networks\Data\Iris_Dataset_setosa.txt'):
    """Read a CSV file and create a list of the values
    :Library: os
    :param path: path of the file to read
    :return: Data : List of lists [item][Atributes]
    """
    if os.path.exists(path) :
        file = open(path)
        Data = [i.replace('\n','').split(',') for i in file.readlines()]
        file.close()
        return Data
    else :
        return None

def sigmoid(x):
    """ The sigmoid function : 1 / (1 + np.exp(-x))
    :param x: float
    :return: image by the sigmoid of x
    """
    return 1 / (1 + np.exp(-x))
def dsigmoid(y):
    """ inverse function of the sigmoid : sigmoid(y) * (1.0 - sigmoid(y))
    :param y: float
    :return: image by the inverse of sigmoid of y
    """
    return sigmoid(y) * (1.0 - sigmoid(y))
def LearningPhase(Data):
    for X in Data:
        X = [float(o) for o in X]
        for i in range(10):
            # Begin of foward progression
            for i in range(len(neu)):  # calculing the neuronnes
                neu[i][0] = X[0] * axe[0][0][i] + X[1] * axe[0][1][i]
                neu[i][1] = sigmoid(neu[i][0])

            Ycalc[0] = neu[0][1] * axe[1][0] + neu[1][1] * axe[1][1] + neu[2][1] * axe[1][2]
            Ycalc[1] = sigmoid(Ycalc[0])
            # End of foward progresion
            # Beining of backward progresion
            error = Y[0] - Ycalc[1]
            deltaOutS = Ycalc[1] * error

            for i in range(len(dweight)):  # adjust axe[1]
                dweight[i] = deltaOutS / neu[i][1]
                axe[1][i] = axe[1][i] + dweight[i]
                dHsum[i] = (deltaOutS / (axe[1][i] - dweight[i])) * dsigmoid(neu[i][0])

            for i in range(len(dweight2)):  # Ajust axe[0]
                for j in range(len(dweight2[i])):
                    dweight2[i][j] = dHsum[j] / X[i]
                    axe[0][i][j] += dweight2[i][j]

            for i in range(len(neu)):  # calculation new neurones
                neu[i][0] = X[0] * axe[0][0][i] + X[1] * axe[0][1][i]
                neu[i][1] = sigmoid(neu[i][0])
                # End of backward progression

            Ycalc[0] = neu[0][1] * axe[1][0] + neu[1][1] * axe[1][1] + neu[2][1] * axe[1][2]
            Ycalc[1] = sigmoid(Ycalc[0])

    return neu, axe

##

#__CORE__

#Creation of var
X = [1,1,1,1]
Y = [1]
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

Data = read('C:\\Users\Lyxilion\Documents\\Neurone\\Neural-Networks\Data\Iris_Dataset_versicolor.txt')
Brain_setosa = LearningPhase(Data)

#__Reset var__
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

Data = read('C:\\Users\Lyxilion\Documents\\Neurone\\Neural-Networks\Data\Iris_Dataset_versicolor.txt')
Brain_virginica = LearningPhase(Data)

#__Reset var__
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

Brain_versicolor = LearningPhase(read('C:\\Users\Lyxilion\Documents\\Neurone\\Neural-Networks\Data\Iris_Dataset_versicolor.txt'))

print(Brain_setosa)
print(Brain_versicolor)
print(Brain_virginica)


#__TESTING PHASE__

SetosaCount = 0
VirginicaCount = 0
VersicolorCount = 0

Data = read('C:\\Users\Lyxilion\Documents\\Neurone\\Neural-Networks\Data\Iris_Dataset_versicolor.txt')
for X in Data :
    X = [float(o) for o in X]

    neu = Brain_virginica[0]
    axe = Brain_virginica[1]

    for i in range(len(neu)):
        neu[i][0] = X[0] * axe[0][0][i] + X[1] * axe[0][1][i]
        neu[i][1] = sigmoid(neu[i][0])

    Ycalc[0] = neu[0][1] * axe[1][0] + neu[1][1] * axe[1][1] + neu[2][1] * axe[1][2]
    Y_virginica = Ycalc[0]

    print("Test virginica: ", Y_virginica)

    neu = Brain_setosa[0]
    axe = Brain_setosa[1]

    for i in range(len(neu)):
        neu[i][0] = X[0] * axe[0][0][i] + X[1] * axe[0][1][i]
        neu[i][1] = sigmoid(neu[i][0])

    Ycalc[0] = neu[0][1] * axe[1][0] + neu[1][1] * axe[1][1] + neu[2][1] * axe[1][2]
    Y_setosa = Ycalc[0]

    print("Test setosa: ", Y_setosa)

    neu = Brain_versicolor[0]
    axe = Brain_versicolor[1]

    for i in range(len(neu)):
        neu[i][0] = X[0] * axe[0][0][i] + X[1] * axe[0][1][i]
        neu[i][1] = sigmoid(neu[i][0])

    Ycalc[0] = neu[0][1] * axe[1][0] + neu[1][1] * axe[1][1] + neu[2][1] * axe[1][2]
    Y_versicolor = Ycalc[0]

    print("Test versicolor: ", Y_versicolor)

    if(Y_setosa > Y_versicolor and Y_setosa > Y_virginica) :
        print("setosa")
        SetosaCount += 1
    if(Y_virginica > Y_setosa and Y_virginica > Y_versicolor) :
        print("virginica")
        VirginicaCount += 1
    if(Y_versicolor > Y_setosa and Y_versicolor > Y_virginica) :
        print("versicolor")
        VersicolorCount += 1

print("setosa", SetosaCount)
print("virginica", VirginicaCount)
print("versicolor", VersicolorCount)