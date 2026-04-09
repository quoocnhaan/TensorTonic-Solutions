import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    y = np.array(y)
    total = len(y)
    
    value, count = np.unique(y, return_counts=True)
    probs = count / total
    entropy = np.dot(probs, np.log2(probs))
    print(probs)
    return -np.sum(entropy)
