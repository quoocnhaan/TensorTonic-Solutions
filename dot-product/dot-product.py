import numpy as np

def dot_product(x, y):
    """
    Compute the dot product of two 1D arrays x and y.
    Must return a float.
    """
    # Write code here
    x = np.array(x)
    y = np.array(y)

    if len(x) != len(y):
        raise ValueError
    
    r = 0
    for i in range(len(x)):
        r += x[i] * y[i] 
    return r