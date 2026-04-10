import numpy as np
def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    # Write code here
    elu = []
    for value in x:
        if value <= 0:
            value = alpha * (np.exp(value) -1)
        elu.append(value)
    return elu