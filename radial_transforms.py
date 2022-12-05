import numpy as np

################################################################
###################RADIAL TRANSFORM FUNCTIONS###################
################################################################


#for normalized versions (000 added to name): divide by (absolute) value (positive-definite anyway) and replace r by a (r_cut)

 ### select in bash script, which feeds it to the programme!

def radial_transform_0(r,factor,a, rad_tr_dis): 
    x = (r-rad_tr_dis) ### #0: NO TRANSFORM
    return x

def radial_transform_1(r,factor,a, rad_tr_dis): # already normalized
    x = a*(1-np.exp(-factor*np.tan(np.pi*(r-rad_tr_dis)/(2*a)))) ### #1:
    return x

def radial_transform_2(r,factor,a,rad_tr_dis):
    x = a*(1-np.exp(-factor*(r-rad_tr_dis))) ### #2: goes to 0 for factor > 1.5 or so, but compression is very strong
    return x

def radial_transform_2000(r,factor,a):
    x = a/np.absolute((1-np.exp(-factor*a)))*(1-np.exp(-factor*r)) ### #2 normalized
    return x

def radial_transform_3(r,factor,a,rad_tr_dis):    
    x = 2.0*a*np.arctan((r-rad_tr_dis)/factor)/np.pi ### #3: looks completely useless, only goes to zero for r -> infinity
    return x

def radial_transform_3000(r,factor,a):    
    x = 2.0*a*np.arctan(r/factor)/np.pi/np.absolute(2*np.arctan(a/factor)/np.pi) ### #3 normalized
    return x
    
def radial_transform_4(r,factor,a,rad_tr_dis):
    x = a*np.tanh((r-rad_tr_dis)/factor) ### #4: try values between 1 and 2
    return x

def radial_transform_4000(r,factor,a):
    x = a/np.absolute(np.tanh(a/factor))*np.tanh(r/factor) ### #4: normalized
    return x

def radial_transform_5(r,factor,a,rad_tr_dis): # already normalized
    x = a*np.tanh(factor*np.tan(np.pi *(r-rad_tr_dis) / (2*a))) ### #5: looks really good
    return x
    
def radial_transform_6(r,factor,a,rad_tr_dis): # already normalized
    x = a*(a/(r-rad_tr_dis))**(-factor) ### #6: factor is actually an exponent now, inspired by ACE math paper (looks shit)
    return x
    
def radial_transform_7(r,factor,a,rad_tr_dis):
    x = a*(1-np.exp(-factor*((r-rad_tr_dis)/a))) ### #7: suggestion 3 from ACE math paper
    return x

def radial_transform_7000(r,factor,a):
    x = a/np.absolute((1-np.exp(-factor*(a/a))))*(1-np.exp(-factor*(r/a))) ### #7 normalized
    return x

def radial_transform_8(r,factor,a,rad_tr_dis):
    x = a*(1+a/(r-rad_tr_dis))**(-factor) ### #8: similar to #6, also inspired by ACE math paper (looks super shit)
    return x

def radial_transform_8000(r,factor,a):
    x = a/np.absolute((1+a/a)**(-factor))*(1+a/r)**(-factor) ### #8 normalized
    return x

def radial_transform_9(r,factor,a,rad_tr_dis):
    x = a*1/(1+np.exp(-factor*(r-rad_tr_dis))) ### #9: sigmoid
    return x

def radial_transform_9999(r,factor,a,rad_tr_dis):
    x = a*1/(1+factor*np.exp(-(r-rad_tr_dis))) ### #9: sigmoid
    return x

def radial_transform_9000(r,factor,a,rad_tr_dis):
    x = a/np.absolute(1/(1+np.exp(-(a-rad_tr_dis))))*1/(1+np.exp(-(r-rad_tr_dis))) ### #9 normalized 
    return x

def radial_transform_10(r,factor,a,rad_tr_dis):
    x = a*np.arctan(np.sinh(factor*(r-rad_tr_dis))) ### #10: Gudermannian
    return x

def radial_transform_1010(r,factor,a,rad_tr_dis):
    x = a*np.arctan(factor*np.sinh((r-rad_tr_dis))) ### #10: Gudermannian
    return x

def radial_transform_10000(r,factor,a,rad_tr_dis):
    x = a/np.absolute(np.arctan(np.sinh(factor*(a-rad_tr_dis))))*np.arctan(np.sinh(factor*(r-rad_tr_dis))) ### #10 normalized
    return x

def radial_transform_11(r,factor,a,rad_tr_dis):
    x = a*np.arctan(factor*(r-rad_tr_dis)) ### #11: simple arctan
    return x

def radial_transform_1111(r,factor,a,rad_tr_dis):
    x = a*factor*np.arctan((r-rad_tr_dis)) ### #11: simple arctan
    return x

def radial_transform_11000(r,factor,a,rad_tr_dis):
    x = a/np.absolute(np.arctan(factor*(a-rad_tr_dis)))*np.arctan(factor*(r-rad_tr_dis)) ### #11 normalized
    return x

def radial_transform_12(r,factor,a,rad_tr_dis):
    x = a*np.tanh(factor*(r-rad_tr_dis)) ### #12: simple tanh
    return x

def radial_transform_1212(r,factor,a,rad_tr_dis):
    x = a*factor*np.tanh((r-rad_tr_dis)) ### #12: simple tanh
    return x

def radial_transform_12000(r,factor,a,rad_tr_dis):
    x = a/np.absolute(np.tanh(factor*(a-rad_tr_dis)))*np.tanh(factor*(r-rad_tr_dis)) ### #12 normalized
    return x


###############################################################
##############RADIAL TRANSFORM SELECTION FUNCTION##############
###############################################################

def select_radial_transform(r, factor, a, rad_tr_dis, rad_tr_selection):
    if rad_tr_selection == 0:
        radial_transform = radial_transform_0(r, factor, a, rad_tr_dis)
    elif rad_tr_selection == 1:
        radial_transform = radial_transform_1(r, factor, a, rad_tr_dis)
    elif rad_tr_selection == 2:
        radial_transform = radial_transform_2(r, factor, a, rad_tr_dis)
    elif rad_tr_selection == 3:
        radial_transform = radial_transform_3(r, factor, a, rad_tr_dis)
    elif rad_tr_selection == 4:
        radial_transform = radial_transform_4(r, factor, a, rad_tr_dis)
    elif rad_tr_selection == 5:
        radial_transform = radial_transform_5(r, factor, a, rad_tr_dis)
    elif rad_tr_selection == 6:
        radial_transform = radial_transform_6(r, factor, a, rad_tr_dis)
    elif rad_tr_selection == 7:
        radial_transform = radial_transform_7(r, factor, a, rad_tr_dis)
    elif rad_tr_selection == 8:
        radial_transform = radial_transform_8(r, factor, a, rad_tr_dis) 
    elif rad_tr_selection == 9:
        radial_transform = radial_transform_9(r, factor, a, rad_tr_dis) 
    elif rad_tr_selection == 10:
        radial_transform = radial_transform_10(r, factor, a, rad_tr_dis) 
    elif rad_tr_selection == 11:
        radial_transform = radial_transform_11(r, factor, a, rad_tr_dis) 
    elif rad_tr_selection == 12:
        radial_transform = radial_transform_12(r, factor, a, rad_tr_dis)     
        ### Questionable transforms:
    elif rad_tr_selection == 9999:
        radial_transform = radial_transform_9999(r, factor, a, rad_tr_dis) 
    elif rad_tr_selection == 1010:
        radial_transform = radial_transform_1010(r, factor, a, rad_tr_dis) 
    elif rad_tr_selection == 1111:
        radial_transform = radial_transform_1111(r, factor, a, rad_tr_dis) 
    elif rad_tr_selection == 1212:
        radial_transform = radial_transform_1212(r, factor, a, rad_tr_dis)  
        ### normalized versions below, names appended by 000
    elif rad_tr_selection == 2000:
        radial_transform = radial_transform_2000(r, factor, a)
    elif rad_tr_selection == 3000:
        radial_transform = radial_transform_3000(r, factor, a)
    elif rad_tr_selection == 4000:
        radial_transform = radial_transform_4000(r, factor, a)
    elif rad_tr_selection == 7000:
        radial_transform = radial_transform_7000(r, factor, a)
    elif rad_tr_selection == 8000:
        radial_transform = radial_transform_8000(r, factor, a)
    ######### Gave up on the idea after trying these #########
    else:
        print('NO MATCHING RADIAL TRANSFORM FOUND')
    return radial_transform