import numpy as np
def debugInitializeWeights(fan_out, fan_in):
    """function who create a matrix with sin"""
   # W = np.reshape(np.sin(W), np.size(W))/10
    W = np.arange(fan_out * (fan_in+1))
    W = W.reshape(fan_out,fan_in+1)
    print (W+1)
    W = (np.sin(W+1))/10
    print (W)

debugInitializeWeights(4,4)