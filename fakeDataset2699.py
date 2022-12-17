#fake dataset

"""
import sklearn
import sklearn.linear_model
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
"""
import matplotlib.pyplot as plt
import random

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
    print(ourList)
    plt.scatter(yList, zList)
    return ourList, yList, zList



#issues with randomizer - do we WANT these to be clustered implicitly for testing purposes?
"""
I'm not entirely sure what we want with our clusters.
We want to run a logistic classifier on them, so I'm assumingw e should run some sort of clf.fit format on them
How did the paper randomize everything?
More importantly, how should we randomize everything? 

"""

"""
def logisticRegressionTime(yMin, yMax, zMin, zMax, amount):
    ourLists = datasetGenerator(yMin, yMax, zMin, zMax, amount)
    clf = sklearn.linear_model.LogisticRegression()
    print(np.array(ourLists[1]), np.array(ourLists[2]))
    clf.fit(np.reshape(np.array(ourLists[1]), -1, 1), np.reshape(np.array(ourLists[2]), -1, 1))
    b = clf.intercept_[0]
    w1, w2 = clf.coef_.T
    c = -b/w2
    m = -w1/w2
    xmin, xmax = yMin, yMax
    ymin, ymax = zMin, zMax
    xd = np.array([xmin, xmax])
    yd = m*xd + c
    plt.plot(xd, yd, 'k', lw=1, ls='--')
    plt.fill_between(xd, yd, ymin, color='tab:blue', alpha=0.2)
    plt.fill_between(xd, yd, ymax, color='tab:orange', alpha=0.2)

    plt.scatter(*X[Y==0].T, s=8, alpha=0.5)
    plt.scatter(*X[Y==1].T, s=8, alpha=0.5)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.ylabel(r'$x_2$')
    plt.xlabel(r'$x_1$')

    plt.show()
"""
