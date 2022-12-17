"""I was rather flustered so I wrote the majority of this out in pseudo-code."""

#2699 baseline code attempt 4
#standard imports

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#part 1

#step 1 - generate

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

#step 3 - draw a line:

#we use kmeans to get an approximate center for our line
def kMeans(ourList1, ourList2):
    neoList = ourList1 + ourList2
    points = np.array(neoList)
    points1 = np.array(ourList1)
    points2 = np.array(ourList2)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(points)
    print(((neoList[0][0], neoList[1][0]), (neoList[0][1], neoList[1][1])))
    return (((neoList[0][0], neoList[1][0]), (neoList[0][1], neoList[1][1])))

#we use this to draw our line. I want to draw a line between the two clusters. 
def lineDrawn(ourList1, ourList2):
    ourPoints = kmeans(ourList1, ourList2)
    a = ourPoints[0]
    b = ourPoints[1]
    #insert a function to get the lines between each.


#questions - how do I want to split everything? Do I want to make my line a random variable between the "true" line so to speak?

#Do I want to assign different odds to each part? example - do I want an 80/20 split above the line and a 20/80 split below the line?

#training shouldn't be too bad of an issue. Probably going to run neural network for this since it's rather easy.

#part 2: false positive testing

def falsePositiveTesting(ourList1, ourList2):
    print("filler")
    #step 1 - append "group 1" label to everything in ourList1. It turns a list of 3-tuples into a list of 4-tuples
    #step 2 - append "group 2" label to everything in ourList2. It turns a list of 3-tuples into a list of 4-tuples.
    #step 3 - independently gather false positive rates for each thing in our list. Compare predicted 1 or 0 to actual 1 or 0.
    #step 4 - do a full sweep of false positive rates. 



def test1(yMin1, yMax1, zMin1, zMax1, amount1, yMin2, yMax2, zMin2, zMax2, amount2):
    list1 = datasetGenerator(yMin1, yMax1, zMin1, zMax1, amount1)[0]
    list2 = datasetGenerator(yMin2, yMax2, zMin2, zMax2, amount2)[0]
    kMeans(list1, list2)
    
test1(2, 3, 4, 5, 200, 2.5, 3.5, 4.5, 5.5, 200)

"""include part 2 as classifying for guilt, not guilty instead of testing for false positive. You can't meaningfully do false positive stuff without predictions.
false positive rate is false positive over TRUE negateive"""
"""only evaluate testing points"""
