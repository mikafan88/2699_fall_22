#2699 baseline code attempt 5.5
#mikafan88

#imports
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split 



#phase 1: data fun

#I. Data Generation

#rewrite dataset generator to talk about which group the thing is from. 
def datasetGenerator(yMin, yMax, zMin, zMax, amount):
    ourList = []
    yList = []
    zList = []
    for i in range(amount):
        y = random.uniform(yMin, yMax)
        z = random.uniform(zMin, zMax)
        ourList.append((y,z))
        yList.append(y)
        zList.append(z)
    return ourList, yList, zList

#II. KMeans to cluster
def kMeans(ourList1, ourList2):
    neoList = ourList1 + ourList2
    points = np.array(neoList)
    points1 = np.array(ourList1)
    points2 = np.array(ourList2)
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(points)
    centroids = kmeans.cluster_centers_
    return (((centroids[0][0], centroids[0][1]), (centroids[1][0], centroids[1][1])))

#III. Draw a line between the centers of the clusters
def mxPlusB(point1, point2): #points 1 and 2 are tuples
    y = 1
    x = 0
    m = (point2[y] - point1[y])/(point2[x] - point1[x])
    b = point1[y] - (point1[x] * m)
    return (m, b)

#IV. Assign ratios based off the line
def trueguilt(ourList1, ourList2, ratio1, ratio2, m, b):
    ourList3 = ourList1 + ourList2
    neoList = []
    for i in range(len(ourList3)):
        a = random.random()
        if ourList3[i][1] > ((m * ourList3[i][0]) + b):
            if a > ratio1:
                neoList.append(ourList3[i] + ("above", ) + (1,)) #guilty
            else:
                neoList.append(ourList3[i] + ("above",) + (0,)) #innocent
        else:
            if a > ratio2:
                neoList.append(ourList3[i] + ("below",) + (1,)) #guilty
            else:
                neoList.append(ourList3[i] + ("below",) + (0,)) #innocent
    
    return neoList #list of 4-tuples

#what's being returned is: (x, y, i {above if i = 1 below if i = 0}, guilty {guilty if i = 1 innocent if i = 0})

#phase 2: determining guilt

def logisticGuiltDeterminer(guiltList):
    guiltArray = np.array(guiltList)
    X = np.delete(guiltArray, 3, 1)
    y = np.delete(guiltArray, 0, 1)
    y = np.delete(y, 0, 1)
    y = np.delete(y, 0, 1)
    Y = np.array(y, dtype = float)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
    TrueX_train = np.array(np.delete(X_train, 2, 1), dtype = float)
    TrueX_test = np.array(np.delete(X_test, 2, 1), dtype = float)
    LR = LinearRegression()
    LR.fit(TrueX_train, np.ravel(y_train))
    y_prediction =  LR.predict(TrueX_test)
    X_test_above_below = np.delete(X_test, 0, 1)
    X_test_above_below = np.delete(X_test_above_below, 0, 1)
    #above, true guilt, predicted guilt
    abomination = list(zip(np.ravel(X_test_above_below), np.array(np.ravel(y_test), dtype = float), np.absolute(np.rint(y_prediction))))
    return abomination

def logisticGuiltDeterminer2(guiltList):
    guiltArray = np.array(guiltList)
    X = np.delete(guiltArray, 3, 1)
    y = np.delete(guiltArray, 0, 1)
    y = np.delete(y, 0, 1)
    y = np.delete(y, 0, 1)
    Y = np.array(y, dtype = float)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
    TrueX_train = np.array(np.delete(X_train, 2, 1), dtype = float)
    TrueX_test = np.array(np.delete(X_test, 2, 1), dtype = float)
    LR = DecisionTreeRegressor(max_depth=5)
    LR.fit(TrueX_train, np.ravel(y_train))
    y_prediction =  LR.predict(TrueX_test)
    X_test_above_below = np.delete(X_test, 0, 1)
    X_test_above_below = np.delete(X_test_above_below, 0, 1)
    #above, true guilt, predicted guilt
    abomination = list(zip(np.ravel(X_test_above_below), np.array(np.ravel(y_test), dtype = float), np.absolute(np.rint(y_prediction))))
    return abomination
    


#phase 3 : checking false positive rate:

def falsePositiveChecker(abomination):
    #split lists
    belowList = []
    aboveList = []
    for i in range(len(abomination)):
        if abomination[i][0] == 'above':
            aboveList.append(abomination[i])
        else:
            belowList.append(abomination[i])
    
    falsePositiveCheckerSupplement(belowList)
    falsePositiveCheckerSupplement(aboveList)

    
def falsePositiveCheckerSupplement(aList): #a supplement to falsepositivechecker
    listSize = len(aList)
    counter = 0
    for i in range(len(aList)):
        if (aList[i][1] == 0) and (aList[i][2] == 1):
            counter = counter + 1
    print(counter / listSize)
    


"""

What I want to do is get my data into the format of two numpy arrays:
an "above" array, and a "below" array. I then want to find the percent of falsely guilty in each.  


"""

#tests
def test1(yMin1, yMax1, zMin1, zMax1, amount1, yMin2, yMax2, zMin2, zMax2, amount2, ratio1, ratio2):
    list1 = datasetGenerator(yMin1, yMax1, zMin1, zMax1, amount1)[0]
    list2 = datasetGenerator(yMin2, yMax2, zMin2, zMax2, amount2)[0]
    points = kMeans(list1, list2)
    mxplusb = mxPlusB(points[0], points[1])
    m = mxplusb[0]
    b = mxplusb[1]
    guiltList = (trueguilt(list1, list2, ratio1, ratio2, m, b))
    abomination = logisticGuiltDeterminer(guiltList)
    abomination2 = logisticGuiltDeterminer2(guiltList)
    falsePositiveChecker(abomination)
    falsePositiveChecker(abomination2)
    return abomination

def test2(yMin1, yMax1, zMin1, zMax1, amount1, yMin2, yMax2, zMin2, zMax2, amount2, ratio1, ratio2):
    list1 = datasetGenerator(yMin1, yMax1, zMin1, zMax1, amount1)[0]
    list2 = datasetGenerator(yMin2, yMax2, zMin2, zMax2, amount2)[0]
    points = kMeans(list1, list2)
    mxplusb = mxPlusB(points[0], points[1])
    m = mxplusb[0]
    b = mxplusb[1]
    guiltList = (trueguilt(list1, list2, ratio1, ratio2, m, b))
    abomination = logisticGuiltDeterminer2(guiltList)
    falsePositiveChecker(abomination)
    return abomination


test1(2, 3, 4, 5, 600, 2.5, 3.5, 4.5, 5.5, 600, .7, .5)

#to do - prototype some sort of regression class
#write up some sort of short document of what we did over the semester. include a table of results.


