# -*- coding: utf-8 -*-

'''
	using kNN on results from a dating site
	created on Jan,11th,2018
	author: wanglei
'''

from numpy import *
import operator
from os import listdir
import datetime

def classify0(inX,dataSet,labels,k):
	dataSetSize = dataSet.shape[0]		#shape[0],first dimension length
	diffMat = tile(inX,(dataSetSize,1)) - dataSet
	sqDiffMat = diffMat**2
	sqDistances = sqDiffMat.sum(axis=1)	#first dimension distances
	distances = sqDistances**0.5
	sortedDistIndicies = distances.argsort()	#argsort(),从小到大排序，返回下标
	classCount = {}
	for i in range(k):							# 这里运行后发现错误，下indicies少个字母i
		votelabel = labels[sortedDistIndicies[i]]	# 对k个样本的所属类别进行统计
		classCount[votelabel] = classCount.get(votelabel,0) + 1	#get()
	sortedClassCount = sorted(classCount.iteritems(),	#iteritems
	 key = operator.itemgetter(1),reverse = True)
	return sortedClassCount[0][0]

def img2vector(filename):
	returnVect = zeros((1,1024))
	fr = open(filename)
	for i in range(32):
		lineStr = fr.readline()
		for j in range(32):
			returnVect[0,32*i+j] = int(lineStr[j])
	return returnVect

def handwritingClassTest():
	starttime = datetime.datetime.now()
	
	hwLabels = []
	trainingFileList = listdir('trainingDigits')
	m = len(trainingFileList)
	trainingMat = zeros((m,1024))
	for i in range(m):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		hwLabels.append(classNumStr)
		trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
	testFileList = listdir('testDigits')
	errorCount = 0.0
	mTest = len(testFileList)
	for i in range(mTest):
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
		classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
		print "the classifier came back with: %d, the real answer is: %d"\
				% (classifierResult,classNumStr)
		if (classifierResult != classNumStr): errorCount += 1.0
	print "\nThe total number of errors is: %d" % errorCount
	print "\nThe total error rate is: %f" % (errorCount/float(mTest))
	
	#long running
	endtime = datetime.datetime.now()
	print "\nRunning time: %s Seconds" %((endtime - starttime).seconds)