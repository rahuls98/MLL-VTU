import numpy as np

#intialize data
x = np.array(([2,9], [1,5], [3,6]), dtype=float)
y = np.array(([92],[86],[89]), dtype=float)

#normalize data
x = x/np.amax(x, axis=0)
y = y/100

#sigmoid function
def sig(x):
    return 1/(1+np.exp(-x))

#sigmoid derivative function
def sigder(x):
    return x*(1-x)

#parameters
epochs = 7000
lr = 0.1

#network structure
iunits = 2
hunits = 3
ounits = 1

whi = np.random.uniform(size=(iunits, hunits)) #input->hidden weights
bh = np.random.uniform(size=(1, hunits)) #hidden layer bias
woh = np.random.uniform(size=(hunits, ounits)) #hidden->output weights
bo = np.random.uniform(size=(1, ounits)) #output layer bias

for i in range(epochs):
    #calulate output of hidden layer by calculating net(h) and applying sigmoid function
    hlin = np.dot(x, whi) #(3x2)*(2x3)
    neth = hlin + bh #(3x3)+(1x3)
    hlout = sig(neth) #(3x3)

    #calculate output of output layer by calculating net(o) and applying sigmoid function
    olin = np.dot(hlout, woh) #(3x3)*(3x1)
    neto = olin + bo #(3x1)+(1x1) added row-wise
    output = sig(neto) #(3x1)

    #calculating del(k) according to algorithm
    """ del(k) = o(k)(1 - o(k))*(t(k)-o(k)) """
    sigder_o = sigder(output) #(3,1)
    error_o = y - output #(3x1)-(3,1)
    del_o = sigder_o * error_o #(3x1)*(3x1)

    #similarly calculating del(h) according to algorithms
    """ del(h) = o(h)(1 - o(h))* summation(h_of_hidden)(w(hk) * del(k)) """
    sigder_h = sigder(hlout) #(3x3)
    contr_h = np.dot(del_o, woh.T) #(3x1)*(1x3)
    del_h = np.dot(sigder_h, contr_h) #(3x3)*(3x3)

    #updating weights according to last step of algorithm
    whi += lr * np.dot(x.T, del_h) #(2,3)*(3x3) = (2x3)
    woh += lr * np.dot(hlout.T, del_o) #(3x3)*(3,1) = (3x1)
    bh += np.sum(del_h, axis=0, keepdims=True) * lr
    bo += np.sum(del_o, axis=0, keepdims=True) * lr #columnwise sum

print("\nInput:\n" + str(x))
print("\nTarget Output:\n" + str(y))
print("\nPredicted Output:\n" + str(output))