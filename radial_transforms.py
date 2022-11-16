import numpy as np

# Radial transform functions
#for normalized versions (000 added to name): divide by (absolute) value (positive-definite anyway) and replace r by a (r_cut)

 ### select in bash script, which feeds it to the programme!

def radial_transform_1(r,factor,a, rad_tr_dis): # already normalized
    #factor1 = 1.2
    x = a*(1-np.exp(-factor*np.tan(np.pi*(r-rad_tr_dis)/(2*a)))) ### #1:
    return x

def radial_transform_2(r,factor,a,rad_tr_dis):
    # factor2 = 0.1
    x = a*(1-np.exp(-factor*(r-rad_tr_dis))) ### #2: goes to 0 for factor > 1.5 or so, but compression is very strong
    return x

def radial_transform_2000(r,factor,a):
    # factor2 = 0.1
    x = a/np.absolute((1-np.exp(-factor*a)))*(1-np.exp(-factor*r)) ### #2 normalized
    return x

def radial_transform_3(r,factor,a,rad_tr_dis):    
    # factor3 = 1.0
    x = 2.0*a*np.arctan((r-rad_tr_dis)/factor)/np.pi ### #3: looks completely useless, only goes to zero for r -> infinity
    return x

def radial_transform_3000(r,factor,a):    
    # factor3 = 1.0
    x = 2.0*a*np.arctan(r/factor)/np.pi/np.absolute(2*np.arctan(a/factor)/np.pi) ### #3 normalized
    return x
    
def radial_transform_4(r,factor,a,rad_tr_dis):
    # factor4 = 2.5
    x = a*np.tanh((r-rad_tr_dis)/factor) ### #4: try values between 1 and 2
    return x

def radial_transform_4000(r,factor,a):
    # factor4 = 2.5
    x = a/np.absolute(np.tanh(a/factor))*np.tanh(r/factor) ### #4: normalized
    return x

def radial_transform_5(r,factor,a,rad_tr_dis): # already normalized
    # factor5 = 2.0
    x = a*np.tanh(factor*np.tan(np.pi *(r-rad_tr_dis) / (2*a))) ### #5: looks really good
    return x
    
def radial_transform_6(r,factor,a,rad_tr_dis): # already normalized
    # factor6 = 0.75
    x = a*(a/(r-rad_tr_dis))**(-factor) ### #6: factor is actually an exponent now, inspired by ACE math paper (looks shit)
    return x
    
def radial_transform_7(r,factor,a,rad_tr_dis):
    # factor7 = 1.5
    x = a*(1-np.exp(-factor*((r-rad_tr_dis)/a))) ### #7: suggestion 3 from ACE math paper
    return x

def radial_transform_7000(r,factor,a):
    # factor7 = 1.5
    x = a/np.absolute((1-np.exp(-factor*(a/a))))*(1-np.exp(-factor*(r/a))) ### #7 normalized
    return x

def radial_transform_8(r,factor,a,rad_tr_dis):
    # factor8 = 0.9
    x = a*(1+a/(r-rad_tr_dis))**(-factor) ### #8: similar to #6, also inspired by ACE math paper (looks super shit)
    #x = a/np.absolute((1+a/a)**(-factor))*(1+a/r)**(-factor)

def radial_transform_8000(r,factor,a):
    # factor8 = 0.9
    x = a/np.absolute((1+a/a)**(-factor))*(1+a/r)**(-factor) ### #8 normalized

    return x