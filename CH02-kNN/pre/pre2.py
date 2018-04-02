# -*- coding: utf-8 -*-

'''
 inX: the input vector to classify
 dataSet: full matrix of training examples
 labels: a vector of labels
 k: the number of nearest neighbors to use in the voting,positive integer
'''

from numpy import *
import operator

def classify0(inX,dataSet,labels,k):
	dataSetSize = dataSet.shape[0]		#shape[0],first dimension length
	diffMat = tile(inX,(dataSetSize,1)) - dataSet
	sqDiffMat = diffMat**2
	sqDistances = sqDiffMat.sum(axis=1)	#first dimension distances
	distances = sqDistances**0.5
	sortedDistIndicies = distances.argsort()	#argsort()
	classCount = {}
	for i in range(k):
		votelabel = labels[sortedDistIndices[i]]
		classCount[votelabel] = classCount.get(votelabel,0) + 1	#get()
	sortedClassCount = sorted(classCount.iteritems(),	#iteritems
	 key = operator.itemgetter(1),reverse = True)
	return sortedClassCount[0][0]