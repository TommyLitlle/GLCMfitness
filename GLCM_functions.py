from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import random
import math
import itertools
import os # new
from time import time
import decimal

   
def get_TM(ls, N, store=False):
    """
    Returns the transition matrix for this landscape. If store=True, it will
    be saved in a field of this object (TM) for later use. If a stored copy already
    exists for this landscape, it will be returned with no wasted computation.
    """
    mut = range(N)                                               # Creates a list (0, 1, ..., N) to call for bitshifting mutations.
    TM = np.zeros((2**N,2**N))                                   # Transition matrix will be sparse (most genotypes unaccessible in one step) so initializes a TM with mostly 0s to do most work for us.
    for i in range(2**N):
        adjMut = [i ^ (1 << m) for m in mut]                     # For the current genotype i, creates list of genotypes that are 1 mutation away.

        adjFit = [ls[j] for j in adjMut]                         # Creates list of fitnesses for each corresponding genotype that is 1 mutation away.
        fitter = list(filter(lambda x: adjFit[x]>ls[i], mut))                      # Finds which indices of adjFit are more fit than the current genotype and thus available for mutation.
        
        fitLen = len(fitter)
        if fitLen == 0:                                          # If no mutations are more fit, stay in current genotype.
            TM[i][i] = 1
        else:
            tranVal = 1.0 / fitLen                                                   # If at least one mutation is more fit, assign a fitness of 1/(# of more fit mutatnts) to each accessible genotype.
            for f in fitter:
                TM[adjMut[f]][i] = tranVal
    return(TM)

def quantize(val, to_values):
    """Quantize a value with regards to a set of allowed values.
    
    Examples:
        quantize(49.513, [0, 45, 90]) -> 45
        quantize(43, [0, 10, 20, 30]) -> 30
    
    Note: function doesn't assume to_values to be sorted and
    iterates over all values (i.e. is rather slow).
    
    Args:
        val        The value to quantize
        to_values  The allowed values
    Returns:
        Closest value among allowed values.
    """
    best_match = None
    best_match_diff = None
    for other_val in to_values:
        diff = abs(other_val - val)
        if best_match is None or diff < best_match_diff:
            best_match = other_val
            best_match_diff = diff
    return best_match

def glcm_landscape_general(lscp, method=1, levels=8, normalise="TRUE"):
    
    N = len(lscp)
    lscp_scale = np.interp(lscp, (min(lscp), max(lscp)), (0, +1))
    
    #Generate quantized values that landscape will be mapped to
    scale = []
    for i in range(levels):
        scale.append(i/(levels-1))
    
    if method==1:
    #Map each value to scaled 
        inds = np.digitize(lscp_scale,scale)
        inds = inds-1
            
    if method==2:
        inds = []
        for x in lscp_scale:    
            y=quantize(x, scale)
            y=int(y*(levels-1))
            inds.append(y)
        
    #print(lscp_scale)
    #print(inds)
    
    #Generate list of neighbours
    hammy = np.array([0,0,0]) 
    for i in range(0,len(lscp)):
        for j in range(i,len(lscp)):
            ob1 = Solution()
            ham = ob1.hammingDistance(i,j)
            if ob1.hammingDistance(i,j)==1:
                hammy=np.vstack([hammy, [i, j, ham]])
    
    #Initialize GLCM        
    hist = np.zeros((len(scale),len(scale)))
    hammy = np.delete(hammy, (0), axis=0)
    #Populate
    for row in hammy:
        x,y,z=row
        hist[inds[x], inds[y]]=hist[inds[x], inds[y]]+1
        
    #Normalized GLCM
    if (normalise=="TRUE"):
        hist = hist/(np.sum(hist))
    
    energy = np.sum(hist*hist)
    
    entropy=0
    contrast=0
    homogeneity=0
    correlation = 0
    mu = 0
    var = 0
    
    for i in range(0, len(scale)):
        for j in range(0, len(scale)):
            mu = mu + i*hist[i,j]

    
    for i in range(0, len(scale)):
        for j in range(0, len(scale)):
            var = var + hist[i,j]*(i-mu)**2
    
    myDict = {}
    for i in range(0, len(scale)):
        for j in range(0, len(scale)):
            entropy = entropy - hist[i,j]*np.log(hist[i,j]+0.00000001)
            contrast = contrast + (i-j)*(i-j)*hist[i,j]
            homogeneity = homogeneity + hist[i,j]/(1+abs(i-j))
            correlation = correlation + hist[i,j]*(i-mu)*(j-mu)/var
            
    N_loci = int(math.log(N,2))
    TM = get_TM(inds,N=N_loci)
    TM_diff = np.sum(abs(get_TM(inds, N=N_loci) - get_TM(lscp, N=N_loci)))
    
    myDict = {"TM":TM, "TM_diff":TM_diff, "levels":levels, "method":method,"energy":energy, "correlation":correlation, "entropy":entropy,"contrast":contrast, "homogeneity":homogeneity}
    #return(energy, entropy, contrast, homogeneity)
    return(myDict)

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

#Bitcount of binary string
def bitcount(n):
    count = 0
    while n > 0:
        count = count + 1
        n = n & (n-1)
    return count


#By Eshan
def int_to_binary(num, pad=4):
    return bin(num)[2:].zfill(pad)

#Single Landscape GLCM function
def glcm_landscape_simple(lscp,  normalise="TRUE"):
    N = len(lscp)
    lscp_scale = np.interp(lscp, (min(lscp), max(lscp)), (0, +1))
    scale = []
    for i in range(N-1):
        scale.append(2*i/N)
    
    #print(N, lscp_scale, scale)

    inds = np.digitize(lscp_scale,scale)
    inds = inds-1
    #print(lscp_scale)
    print(inds)
    hammy = np.array([0,0,0]) #generate neighbours
    for i in range(0,len(lscp)):
        for j in range(i,len(lscp)):
            ob1 = Solution()
            ham = ob1.hammingDistance(i,j)
            if ob1.hammingDistance(i,j)==1:
                hammy=np.vstack([hammy, [i, j, ham]])
            
    #print(hammy)
    hammy = np.delete(hammy,(0),axis=0)
    #print(hammy)

    hist = np.zeros((len(scale),len(scale)))

    for row in hammy:
        x,y,z=row
        hist[inds[x], inds[y]]=hist[inds[x], inds[y]]+1
    
    if (normalise=="TRUE"):
        hist = hist/(np.sum(hist))
    
    #print(hist)
    energy = np.sum(hist*hist)
    entropy=0
    contrast=0
    homogeneity=0
    
    correlation = 0
    mu = 0
    var = 0
    
    for i in range(0, len(scale)):
        for j in range(0, len(scale)):
            mu = mu + i*hist[i,j]

    
    for i in range(0, len(scale)):
        for j in range(0, len(scale)):
            var = var + hist[i,j]*(i-mu)**2
    
    myDict = {}
    for i in range(0, len(scale)):
        for j in range(0, len(scale)):
            entropy = entropy - hist[i,j]*np.log(hist[i,j]+0.00000001)
            contrast = contrast + (i-j)*(i-j)*hist[i,j]
            homogeneity = homogeneity + hist[i,j]/(1+abs(i-j))
            correlation = correlation + hist[i,j]*(i-mu)*(j-mu)/var
    
    N_loci = int(math.log(N,2))
    TM = get_TM(inds,N=N_loci)
    TM_diff = np.sum(abs(get_TM(inds, N=N_loci) - get_TM(lscp, N=N_loci)))
 
    
    myDict = {"TM":TM, "TM_diff":TM_diff, "energy":energy, "correlation":correlation, "entropy":entropy,"contrast":contrast, "homogeneity":homogeneity, "glcm":hist}
    #return(energy, entropy, contrast, homogeneity)
    return(myDict)

class Solution(object):
   def hammingDistance(self, x, y):
      """
      :type x: int
      :type y: int
      :rtype: int
      """
      ans = 0
      for i in range(31,-1,-1):
         b1= x>>i&1
         b2 = y>>i&1
         ans+= not(b1==b2)
         #if not(b1==b2):
            # print(b1,b2,i)
      return ans


#Calculate GLCM between pairs of same genotypes on different landscapes

def glcm_landscape_between(lscp1, lscp2, normalise="TRUE"):
    
    N1 = len(lscp1)
    lscp_scale1 = np.interp(lscp1, (min(lscp1), max(lscp1)), (0, +1))
    scale1 = []
    for i in range(8):
        scale1.append(2*i/N1)
        
    N2 = len(lscp2)
    lscp_scale2 = np.interp(lscp2, (min(lscp2), max(lscp2)), (0, +1))
    scale2 = []
    for i in range(8):
        scale2.append(2*i/N2)
    
    #print(lscp1, lscp2, lscp_scale1, lscp_scale2)

    inds1 = np.digitize(lscp_scale1,scale1)
    inds1 = inds1-1
    
    inds2 = np.digitize(lscp_scale2,scale2)
    inds2 = inds2-1
    
    hammy = np.array([0,0,0]) #generate neighbours
    for i in range(0,len(lscp1)):
        hammy=np.vstack([hammy, [i,i,1]])
        
        #for j in range(i,len(lscp1)):
        #    ob1 = Solution()
        #   ham = ob1.hammingDistance(i,j)
        #   if ob1.hammingDistance(i,j)==1:
        #        hammy=np.vstack([hammy1, [i, j, ham]])
            
    #print(hammy)
    hammy = np.delete(hammy, (0), axis=0)

    #print(inds1, inds2)
    
    hist = np.zeros((len(scale1),len(scale1)))

    for row in hammy:
        x,y,z=row
        print(inds1[x], inds2[y])
        hist[inds1[x], inds2[y]]=hist[inds1[x], inds2[y]]+1
    
    if (normalise=="TRUE"):
        hist = hist/(np.sum(hist))
    
    #print(hist)
    energy = np.sum(hist*hist)
    entropy=0
    contrast=0
    homogeneity=0
       
    correlation = 0
    mui = 0
    muj = 0
    vari = 0
    varj = 0
    
    for i in range(0, len(scale1)):
        for j in range(0, len(scale1)):
            mui = mui + i*hist[i,j]
            muj = muj + j*hist[i,j]
    print(mui, muj)
    
    for i in range(0, len(scale1)):
        for j in range(0, len(scale1)):
            vari = vari + hist[i,j]*(i-mui)**2
            varj = varj + hist[i,j]*(j-muj)**2
    
    myDict = {}
    for i in range(0, len(scale1)):
        for j in range(0, len(scale1)):
            entropy = entropy - hist[i,j]*np.log(hist[i,j]+0.00000001)
            contrast = contrast + (i-j)*(i-j)*hist[i,j]
            homogeneity = homogeneity + hist[i,j]/(1+abs(i-j))
            correlation = correlation + hist[i,j]*(i-mui)*(j-muj)/math.sqrt(vari*varj)
 
    
    myDict = {"energy":energy, "correlation":correlation, "entropy":entropy,"contrast":contrast, "homogeneity":homogeneity}
    #return(energy, entropy, contrast, homogeneity)
    
    return(myDict)




def glcm_landscape_scale(key, lscp, drug_group, scale):
    lscp_scale = lscp
    inds = np.digitize(lscp_scale,scale)
    inds = inds-1
    #peaks = sum(lscp_scale==max(lscp_scale))
    peaks = 0

    
    hammy = np.array([0,0,0])
    for i in range(0,len(lscp)):
        for j in range(i,len(lscp)):
            ob1 = Solution()
            ham = ob1.hammingDistance(i,j)
            if ob1.hammingDistance(i,j)==1:
                hammy=np.vstack([hammy, [i, j, ham]])
            
    #print(hammy)

    hist = np.zeros((len(scale),len(scale)))

    for row in hammy:
        x,y,z=row
        hist[inds[x], inds[y]]=hist[inds[x], inds[y]]+1
        
    hist = hist/(np.sum(hist))

    energy = np.sum(hist*hist)
    entropy=0
    contrast=0
    homogeneity=0
    correlation = 0
    mu = 0
    var = 0
    
    for i in range(0, len(scale)):
        for j in range(0, len(scale)):
            mu = mu + i*hist[i,j]

    
    for i in range(0, len(scale)):
        for j in range(0, len(scale)):
            var = var + hist[i,j]*(i-mu)**2
    
    myDict = {}
    for i in range(0, len(scale)):
        for j in range(0, len(scale)):
            entropy = entropy - hist[i,j]*np.log(hist[i,j]+0.00000001)
            contrast = contrast + (i-j)*(i-j)*hist[i,j]
            homogeneity = homogeneity + hist[i,j]/(1+abs(i-j))
            correlation = correlation + hist[i,j]*(i-mu)*(j-mu)/var
    
    myDict = {"mu":mu, "var":var,"peaks":peaks,"correlation":correlation,"group":drug_group[key],"energy":energy, "entropy":entropy,"contrast":contrast, "homogeneity":homogeneity}
    #return(energy, entropy, contrast, homogeneity)
    return(myDict)
       
    
 