#===========================================================================================#
import numpy as np
from math import log
from copy import deepcopy
from random import random
from random import sample
from itertools import chain

#===========================================================================================#
# Helper Functions
#===========================================================================================#

# Takes a genotype and converts it to an integer for use indexing the fitness landscape list 
def convertGenotypeToInt(genotype):
	out = 0
	for bit in genotype:
		out = (out << 1) | bit
	return out

# Converts an integer to a genotype by taking the binary value and padding to the left by 0s		
def convertIntToGenotype(anInt, pad):
	offset = 2**pad
	return [int(x) for x in bin(offset+anInt)[3:]]	

# Function which returns all genotypes at Hamming distance 1 from a specified genotype
def getOneStepNeighbours(genotype):
	neighbours = []
	for x in range(0, len(genotype)):
		temp = deepcopy(genotype)
		temp[x] = (genotype[x]+1) %2 #There is some inefficiency here.
		neighbours.append(temp)
	return neighbours

#===========================================================================================#

# Landscapes
#===========================================================================================#

class FitnessLandscape:
	def __init__(self, landscapeValues, name=None):
		self.landscape = landscapeValues
		self.name = name

	def getFitness(self, genotype):
		fitness = self.landscape[convertGenotypeToInt(genotype)]
		return fitness

	def genotypeLength(self):
		return int(log(len(self.landscape), 2))

	def numGenotypes(self):
		return len(self.landscape)

	def isPeak(self, g):
		peak = True
		for h in getOneStepNeighbours(g):
			if self.getFitness(g) < self.getFitness(h):
				peak = False
				break
		return peak

	def getPeaks(self):
		peaks = []

		allGenotypes = []
		N =self.genotypeLength()
		for x in range(0, 2**N):
			allGenotypes.append(convertIntToGenotype(x, self.genotypeLength()))

		for g in allGenotypes:
			if self.isPeak(g):
				peaks.append(g)
		
		return peaks

	def getGlobalPeak(self):
		return convertIntToGenotype(np.argmax(self.landscape), self.genotypeLength())

	def getLowestFitnessPeak(self):
		# Finds the peaks of the landscape
		peak_genotypes = self.getPeaks()
		lowest_peak_genotype = peak_genotypes[np.argmin([self.getFitness(g) for g in peak_genotypes])]
		return lowest_peak_genotype



#===========================================================================================#
# An implementation of Kauffman's NK models for generating tunably rugged landscapes
#===========================================================================================#

# For each possible value of (x_i;x_{i_1}, ... , x_{i_K}) we have a random number sampled from [0,1)
def geneWeights(K,N):
	return [[random() for x in range(2**(K+1))] for y in range(N)]

# Given a genotype length (N) and the number of alleles (K) this function randomly choses K alleles 
# from positions {1,...N}\{i} which interact epistatically with the ith position	
def buildInfringersTable(N,K):
	return [sample(list(chain(range(i),range(i+1,N))),K) for i in range(N)]

# Builds a tuple for look up in the fitness table from the infringers list
def buildTuple(allele, i, infringers):
	tp = [allele[i]]
	for j in infringers:
		tp += [allele[j]]
	return tp

# Given an allele computes the fitness by building a tuple of revelant infringers
# and looks them up in the gene weights table
def alleleFitness(allele, gw, infrs, N, K):
	s = 0.
	for i in range(N):
		index = buildTuple(allele,i,infrs[i])
		s += gw[i][convertGenotypeToInt(index)]
	s = s / N
	return s

#Generates an N-K landscape from Kauffman's method.
def generateNKLandscape(N,K):
	gw = geneWeights(K,N)
	infrs = buildInfringersTable(N,K)
	landscape = [alleleFitness(convertIntToGenotype(a,N), gw, infrs, N, K) for a in range(2**N)]
	return(landscape)
    #return FitnessLandscape(landscape)

def generateNKLandscape_FL(N,K):
	gw = geneWeights(K,N)
	infrs = buildInfringersTable(N,K)
	landscape = [alleleFitness(convertIntToGenotype(a,N), gw, infrs, N, K) for a in range(2**N)]
	return FitnessLandscape(landscape)
    #return FitnessLandscape(landscape)
	