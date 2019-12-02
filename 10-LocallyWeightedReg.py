import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def kernel(point, xmat, k):
    m = np.shape(xmat)[0]
    weights = np.mat(np.eye((m)))
    for j in range(m):
        diff = point - xmat[j]
        weights[j,j] = np.exp((diff*diff.T)/(-2.0*(k**2)))

    return weights

def lw(point, xmat, ymat, k):
    weights = kernel(point, xmat, k)
    W = (x.T*(weights*x)).I*(x.T*(weights*ymat.T))
    return W

def lwr(xmat, ymat, k):
    m = np.shape(xmat)[0]
    ypred = np.zeros(m)
    for i in range(m):
        ypred[i] = xmat[i] * lw(xmat[i], xmat, ymat, k)

    return ypred

def plot(x, ypred, bill, tip):
    sortindex = x[:,1].argsort(0)
    xsort = x[sortindex][:,0]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.scatter(bill, tip, color='green')
    ax.plot(xsort[:,1], ypred[sortindex], color='red', linewidth=3)

    plt.xlabel('Total Bill')
    plt.ylabel('Tip')

    plt.show()



data = pd.read_csv('datasets/10.csv')
bill = np.array(data.total_bill)
tip = np.array(data.tip)
mbill = np.mat(bill)
mtip = np.mat(tip)
m = np.shape(mbill)[1]
one = np.mat(np.ones(m))
x = np.hstack((one.T, mbill.T))
ypred = lwr(x, mtip, 0.5)
plot(x, ypred, bill, tip)