import numpy as np

# Radial transform functions

 ### get this from input file!

def radial_transform_1(r,factor,a):
    # Function that defines the radial transform x = xi(r).
    #factor1 = 1.2
    x = a*(1-np.exp(-factor*np.tan(np.pi*r/(2*a)))) ### #1: goes to 0
    return x

def radial_transform_2(r,factor,a):

    # factor2 = 0.1
    x = a*(1-np.exp(-factor*r)) ### #2: goes to 0 for factor > 1.5 or so, but compression is very strong
    return x
    
def radial_transform_3(r,factor,a):    
    # factor3 = 1.0
    x = 2.0*a*np.arctan(r/factor)/np.pi ### #3: looks completely useless, only goes to zero for r -> infinity
    return x
    
def radial_transform_4(r,factor,a):
    # factor4 = 2.5
    x = a*np.tanh(r/factor) ### #4: try values between 1 and 2
    return x
    
def radial_transform_5(r,factor,a):    
    # factor5 = 2.0
    x = a*np.tanh(factor*np.tan(np.pi *r / (2*a))) ### #5: looks really good
    return x
    
def radial_transform_6(r,factor,a):
    # factor6 = 0.75
    x = a*(a/r)**(-factor) ### #6: factor is actually an exponent now, inspired by ACE math paper (looks shit)
    return x
    
def radial_transform_7(r,factor,a):
    # factor7 = 1.5
    x = a*(1-np.exp(-factor*(r/a))) ### #7: suggestion 3 from ACE math paper
    return x
    
def radial_transform_8(r,factor,a):
    # factor8 = 0.9
    x = a*(1+a/r)**(-factor) ### #8: similar to #6, also inspired by ACE math paper (looks super shit)
    #x = a/np.absolute((1+a/a)**(-factor))*(1+a/r)**(-factor)

    return x