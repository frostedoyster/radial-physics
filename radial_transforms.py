import numpy as np

################################################################
###################RADIAL TRANSFORM FUNCTIONS###################
################################################################

#for normalized versions (000 added to name): divide by (absolute) value (positive-definite anyway) and replace r by a (r_cut)

 ### select in bash script, which feeds it to the programme!

def radial_transform_0000(r,factor,a, rad_tr_dis): ### #0: For Testing!
    x = (1+r/a)^(-factor) # Drautz favorite
    return x

def radial_transform_0(r,factor,a, rad_tr_dis): 
    x = (r-rad_tr_dis) ### #0: NO TRANSFORM
    return x

def radial_transform_1(r,factor,a, rad_tr_dis): # already normalized
    x = a*(1-np.exp(-factor*np.tan(np.pi*(r-rad_tr_dis)/(2*a)))) ### #1:
    return x

def radial_transform_2(r,factor,a,rad_tr_dis):
    x = a*(1-np.exp(-factor*(r-rad_tr_dis))) ### #2: goes to 0 for factor > 1.5 or so, but compression is very strong
    return x

def radial_transform_2222(r,factor,a,rad_tr_dis):
    x = a*(1-np.exp(-(r-rad_tr_dis)/factor)) ### #2222: same as #2 just w. factor in denominator
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
    x = a*(a/(r-rad_tr_dis))**(-factor) ### #6: factor is actually an exponent now, inspired (not suggested!) by ACE math paper (looks shit)
    return x
    
def radial_transform_7(r,factor,a,rad_tr_dis):
    x = a*(1-np.exp(-factor*((r-rad_tr_dis)/a))) ### #7: inspired by ACE math paper
    return x

def radial_transform_7000(r,factor,a):
    x = a/np.absolute((1-np.exp(-factor*(a/a))))*(1-np.exp(-factor*(r/a))) ### #7 normalized
    return x

def radial_transform_8(r,factor,a,rad_tr_dis):
    x = a*(1+a/(r-rad_tr_dis))**(-factor) ### #8: similar to #6, also inspired (not suggested!) by ACE math paper (looks super shit)
    return x

def radial_transform_8000(r,factor,a):
    x = a/np.absolute((1+a/a)**(-factor))*(1+a/r)**(-factor) ### #8 normalized
    return x

def radial_transform_9(r,factor,a,rad_tr_dis):
    x = a*1/(1+np.exp(-factor*(r-rad_tr_dis))) ### #9: sigmoid
    return x

def radial_transform_9000(r,factor,a,rad_tr_dis):
    x = a/np.absolute(1/(1+np.exp(-(a-rad_tr_dis))))*1/(1+np.exp(-(r-rad_tr_dis))) ### #9 normalized 
    return x

def radial_transform_10(r,factor,a,rad_tr_dis):
    x = a*np.arctan(np.sinh(factor*(r-rad_tr_dis))) ### #10: Gudermannian
    return x

def radial_transform_10000(r,factor,a,rad_tr_dis):
    x = a/np.absolute(np.arctan(np.sinh(factor*(a-rad_tr_dis))))*np.arctan(np.sinh(factor*(r-rad_tr_dis))) ### #10 normalized
    return x

def radial_transform_11(r,factor,a,rad_tr_dis):
    x = a*np.arctan(factor*(r-rad_tr_dis)) ### #11: simple arctan
    return x

def radial_transform_11000(r,factor,a,rad_tr_dis):
    x = a/np.absolute(np.arctan(factor*(a-rad_tr_dis)))*np.arctan(factor*(r-rad_tr_dis)) ### #11 normalized
    return x

def radial_transform_12(r,factor,a,rad_tr_dis):
    x = a*np.tanh(np.sinh(factor*(r-rad_tr_dis))) ### #12: simple tanh
    return x

def radial_transform_12000(r,factor,a,rad_tr_dis):
    x = a/np.absolute(np.tanh(factor*(a-rad_tr_dis)))*np.tanh(factor*(r-rad_tr_dis)) ### #12 normalized
    return x

def radial_transform_13(r,factor,a,rad_tr_dis):
    x = a*(1-np.exp(-factor*(r-rad_tr_dis)**2))  ### #13 only r squared
    return x 

def radial_transform_14(r,factor,a,rad_tr_dis):
    x = a*(1-np.exp(-factor*(r-rad_tr_dis)**2))*(1-np.exp(-factor*(r-rad_tr_dis)))  ### #14 The 'nice' S shaped exponential
    return x 

def radial_transform_15(r,factor,a,rad_tr_dis):
    x = a*(1-np.exp(-factor*(r-rad_tr_dis)**2-factor*(r-rad_tr_dis))) ### #15 The less 'nice' S shaped exponential
    return x

def radial_transform_16(r,factor,a,rad_tr_dis):
    x = a*(1-np.exp(-1.5*factor*(r-rad_tr_dis)**2))*(1-np.exp(-factor*(r-rad_tr_dis))) ### #16 Variation of #14 w different factor ratio
    return x

def radial_transform_17(r,factor,a,rad_tr_dis):
    x = a*(1-np.exp(-factor*(r-rad_tr_dis)**2))*(1-np.exp(-0.7*factor*(r-rad_tr_dis)))  ### #17 Variation of #14 w different factor ratio
    return x

def radial_transform_18(r,factor,a,rad_tr_dis):
    x = a*(1-np.exp(-2*factor*(r-rad_tr_dis)**2))*(1-np.exp(-factor*(r-rad_tr_dis))) ### #18 Variation of #14 w different factor ratio
    return x

def radial_transform_19(r,factor,a,rad_tr_dis):
    x = a*(1-np.exp(-3*factor*(r-rad_tr_dis)**2))*(1-np.exp(-factor*(r-rad_tr_dis)))  ### #19 Variation of #14 w different factor ratio
    return x

def radial_transform_20(r,factor,a,rad_tr_dis):
    x = a*(1-np.exp(-5*factor*(r-rad_tr_dis)**2))*(1-np.exp(-factor*(r-rad_tr_dis)))  ### #20 Variation of #14 w different factor ratio
    return x

def radial_transform_21(r,factor,a,rad_tr_dis):
    x = a*(1-np.exp(-10*factor*(r-rad_tr_dis)**2))*(1-np.exp(-factor*(r-rad_tr_dis)))  ### #21 Variation of #14 w different factor ratio
    return x

def radial_transform_22(r,factor,a,rad_tr_dis):
    x = a*(1-np.exp(-factor*(r-rad_tr_dis)**1.5))*(1-np.exp(-factor*(r-rad_tr_dis)))  ### #22 Variation of #14 w different exponent
    return x 

def radial_transform_23(r,factor,a,rad_tr_dis):
    x = a*(1-np.exp(-factor*(r-rad_tr_dis)**2.5))*(1-np.exp(-factor*(r-rad_tr_dis)))  ### #23 Variation of #14 w different exponent
    return x

def radial_transform_24(r,factor,a,rad_tr_dis):
    x = a*(1-np.exp(-factor*(r-rad_tr_dis)**3))*(1-np.exp(-factor*(r-rad_tr_dis)))  ### #24 Variation of #14 w different exponent
    return x 

def radial_transform_25(r,factor,a,rad_tr_dis):
    x = a*(1-np.exp(-2*factor*(r-rad_tr_dis)**2.5))*(1-np.exp(-factor*(r-rad_tr_dis)))  ### #25 Variation of #14 w different exponent and factor ratio
    return x 

def radial_transform_26(r,factor,a,rad_tr_dis):
    x = a*(1-np.exp(-20*factor*(r-rad_tr_dis)**2))*(1-np.exp(-factor*(r-rad_tr_dis)))  ### #26 Variation of #14 w different factor ratio
    return x 

def radial_transform_27(r,factor,a,rad_tr_dis):
    x = a*(1-np.exp(-30*factor*(r-rad_tr_dis)**2))*(1-np.exp(-factor*(r-rad_tr_dis)))  ### #27 Variation of #14 w different factor ratio
    return x 

def radial_transform_28(r,factor,a,rad_tr_dis):
    x = a*(1-np.exp(-(r-rad_tr_dis)**1.5/(factor**2)))*(1-np.exp(-1.4*(r-rad_tr_dis)))  ### #28 for testing
    #x = a*(1-np.exp(-30*factor*(r-rad_tr_dis)**2))*(1-np.exp(-factor*(r-rad_tr_dis)))
    return x 

###############################################################
##############RADIAL TRANSFORM SELECTION FUNCTION##############
###############################################################

def select_radial_transform(r, factor, a, rad_tr_dis, rad_tr_selection):
    if rad_tr_selection == 0:
        radial_transform = radial_transform_0(r, factor, a, rad_tr_dis)
    elif rad_tr_selection == 0000:
        radial_transform = radial_transform_0000(r, factor, a, rad_tr_dis)
    elif rad_tr_selection == 1:
        radial_transform = radial_transform_1(r, factor, a, rad_tr_dis)
    elif rad_tr_selection == 2:
        radial_transform = radial_transform_2(r, factor, a, rad_tr_dis)
    elif rad_tr_selection == 2222:
        radial_transform = radial_transform_2222(r, factor, a, rad_tr_dis)
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
    elif rad_tr_selection == 13:
        radial_transform = radial_transform_13(r, factor, a, rad_tr_dis)   
    elif rad_tr_selection == 14:
        radial_transform = radial_transform_14(r, factor, a, rad_tr_dis)   
    elif rad_tr_selection == 15:
        radial_transform = radial_transform_15(r, factor, a, rad_tr_dis)
    elif rad_tr_selection == 16:
        radial_transform = radial_transform_16(r, factor, a, rad_tr_dis)
    elif rad_tr_selection == 17:
        radial_transform = radial_transform_17(r, factor, a, rad_tr_dis)  
    elif rad_tr_selection == 18:
        radial_transform = radial_transform_18(r, factor, a, rad_tr_dis)
    elif rad_tr_selection == 19:
        radial_transform = radial_transform_19(r, factor, a, rad_tr_dis)
    elif rad_tr_selection == 20:
        radial_transform = radial_transform_20(r, factor, a, rad_tr_dis)   
    elif rad_tr_selection == 21:
        radial_transform = radial_transform_21(r, factor, a, rad_tr_dis)  
    elif rad_tr_selection == 22:
        radial_transform = radial_transform_22(r, factor, a, rad_tr_dis)
    elif rad_tr_selection == 23:
        radial_transform = radial_transform_23(r, factor, a, rad_tr_dis)   
    elif rad_tr_selection == 24:
        radial_transform = radial_transform_24(r, factor, a, rad_tr_dis) 
    elif rad_tr_selection == 25:
        radial_transform = radial_transform_25(r, factor, a, rad_tr_dis) 
    elif rad_tr_selection == 26:
        radial_transform = radial_transform_26(r, factor, a, rad_tr_dis) 
    elif rad_tr_selection == 27:
        radial_transform = radial_transform_27(r, factor, a, rad_tr_dis) 
    elif rad_tr_selection == 28:
        radial_transform = radial_transform_28(r, factor, a, rad_tr_dis) 
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