# Basic back propagation demo
Intended to follow as close as possible the python code by Andrew Trask in the following blog post: https://iamtrask.github.io/2015/07/12/basic-python-network/

I've broken up the code a little to make it somewhat more readable.

I've also added a parameter for learning rate.

This implementation uses the ND4J linear algebra library

## Andrew Trask's python implementation
    X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
    y = np.array([[0,1,1,0]]).T
    syn0 = 2*np.random.random((3,4)) - 1
    syn1 = 2*np.random.random((4,1)) - 1
    for j in xrange(60000):
        l1 = 1/(1+np.exp(-(np.dot(X,syn0))))
        l2 = 1/(1+np.exp(-(np.dot(l1,syn1))))
        l2_delta = (y - l2)*(l2*(1-l2))
        l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
        syn1 += l1.T.dot(l2_delta)
        syn0 += X.T.dot(l1_delta)