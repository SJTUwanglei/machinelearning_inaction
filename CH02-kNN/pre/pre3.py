# -*- coding: utf-8 -*-
# 返回的是前三列数据和最后一列非字符串的分类的数字类1，2，3

from numpy import *
import operator
from collections import Counter	#注意这句引用


def file2matrix(filename):
	fr = open(filename)
	numberOfLines = len(fr.readlines())		# know lines
	returnMat = zeros((numberOfLines,3))	# create a Numpy matrix
	classLabelVector = []
	index = 0
	for line in fr.readlines():
		line = line.strip()
		listFromLine = line.split('\t')		# return list
		returnMat[index,:] = listFromLine[0:3]
		classLabelVector.append(listFromLine[-1])	# 这里注意转化问题
		index += 1
	
	#将列表的最后一列由字符串转化为数字以便计算，用dataTestSet的txt
	dictClassLabel = Counter(classLabelVector)	#加入了collections里的Counter
	classLabel = []
	kind = list(dictClassLabel)
	for item in classLabelVector:
		if item == kind[0]:
			item = 1
		elif item == kind[1]:
			item = 2
		else:
			item = 3
		classLabel.append(item)
	return returnMat,classLabel
	


