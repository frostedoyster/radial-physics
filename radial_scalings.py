import numpy as np

def radial_scaling(factor, r): 
    x = (1-np.exp(-factor*r))*np.exp(-factor*r)
    return x