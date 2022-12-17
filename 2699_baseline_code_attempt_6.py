#2699 baseline code attempt 6
#mikafan88

#imports
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split 



#phase 1: data fun

#I. Data Generation

#rewrite dataset generator to talk about which group the thing is from. 
def datasetGenerator(yMin, yMax, zMin, zMax, amount, label):
    ourList = []
    yList = []
    zList = []
    labeledList = []
    for i in range(amount):
        y = random.uniform(yMin, yMax)
        z = random.uniform(zMin, zMax)
        ourList.append((y,z))
        labeledList.append((y, z, label))
        yList.append(y)
        zList.append(z)
    return ourList, yList, zList, labeledList

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
                neoList.append(ourList3[i] + (1,)) #guilty
            else:
                neoList.append(ourList3[i] + (0,)) #innocent
        else:
            if a > ratio2:
                neoList.append(ourList3[i] + (1,)) #guilty
            else:
                neoList.append(ourList3[i] + (0,)) #innocent
    
    return neoList #list of 4-tuples

#what's being returned is: (x, y, i {above if i = 1 below if i = 0}, guilty {guilty if i = 1 innocent if i = 0})

#phase 2: determining guilt

def logisticGuiltDeterminer(guiltList):
    guiltArray = np.array(guiltList)
    X = np.delete(guiltArray, 3, 1)
    #X = np.delete(X, 2, 1)
    #X = np.array(X, dtype = float)
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
    print(np.absolute(np.rint(y_prediction)))
    X_test_above_below = np.delete(X_test, 0, 1)
    X_test_above_below = np.delete(X_test_above_below, 0, 1)
    print(np.ravel(X_test_above_below))
    print(np.array(np.ravel(y_test), dtype = float))
    return guiltArray
    

#phase 3 : checking false positive rate:


"""

What I want to do is get my data into the format of two numpy arrays:
an "above" array, and a "below" array. I then want to find the percent of falsely guilty in each.  


"""

#tests
def test1(yMin1, yMax1, zMin1, zMax1, amount1, yMin2, yMax2, zMin2, zMax2, amount2, ratio1, ratio2, label1, label2):
    dataset1 = datasetGenerator(yMin1, yMax1, zMin1, zMax1, amount1, label1)[0]
    list1 = dataset1[0]
    labeledList1 = dataset1[3]
    dataset2 = datasetGenerator(yMin2, yMax2, zMin2, zMax2, amount2, label2)[0]
    list2 = dataset2[0]
    labeledList2 = dataset2[3]
    points = kMeans(list1, list2)
    mxplusb = mxPlusB(points[0], points[1])
    m = mxplusb[0]
    b = mxplusb[1]
    guiltList = (trueguilt(labeledList1, labeledList2, ratio1, ratio2, m, b))
    guiltArray = logisticGuiltDeterminer(guiltList)
    return guiltArray

test1(2, 3, 4, 5, 200, 2.5, 3.5, 4.5, 5.5, 200, .999, .001, "b", "w")


#look into RFs for paper
#look into how to fit reinforced learnin
#my approach to this would be focused around RFs and bayesian

#step 1 - bayesian classify
#step 2 - rf

#papers on logistic classifiers 

#make numbers less obvious. see if you can output some sort of table.
#treat squares as two different groups. make square 1 and square 2 my categories. 
#false positive testing needs to be changed to testing groups and not above/below
#need to find a way to scale this up.
#class for actual model itself

#11/9/22
#tasks for next week
# - code rewritten to handle 3+ groups
# - fix my false positive checker
# - start building a model. look at scikit data. 
