# -*- coding: utf-8 -*- 
# 程序最后一个分类器出不了结果

from math import log
import operator
import matplotlib.pyplot as plt

def calcShannonEnt(dataSet):
	numEntries = len(dataSet)
	labelCounts = {}
	for feaVec in dataSet:
		currentLabel = feaVec[-1]
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1  # 计数的记错了
	shannonEnt = 0.0
	for key in labelCounts:
		prob = float(labelCounts[key])/numEntries
		shannonEnt -= prob * log(prob,2)
	return shannonEnt

def createDataSet():
	dataSet = [[1,1,'yes'],
			   [1,1,'yes'],
			   [1,0,'no'],
			   [0,1,'no'],
			   [0,1,'no']]
	labels = ['no surfacing','flippers']
	return dataSet, labels

def splitDataSet(dataSet, axis, value):
	retDataSet = []
	for feaVec in dataSet:
		if feaVec[axis] == value:
			reducedFeaVec = feaVec[:axis]
			reducedFeaVec.extend(feaVec[axis+1:])
			retDataSet.append(reducedFeaVec)
	return retDataSet

def chooseBestFeatureToSplit(dataSet):
	numFeatures = len(dataSet[0]) - 1
	baseEntropy = calcShannonEnt(dataSet)
	bestInfoGain = 0.0; bestFeature = -1
	for i in range(numFeatures):
		feaList = [example[i] for example in dataSet]
		uniqueVals = set(feaList)
		newEntropy = 0.0
		for value in uniqueVals:
			subDataSet = splitDataSet(dataSet, i, value)
			prob = len(subDataSet)/float(len(dataSet))
			newEntropy += prob * calcShannonEnt(subDataSet)
		infoGain = baseEntropy - newEntropy
		if (infoGain > bestInfoGain):
			bestInfoGain = infoGain
			bestFeature = i
	return bestFeature

def majorityCnt(classList):
	classCount = {}
	for vote in classList:
		if vote not in classCount.keys(): classCount[vote] = 0
		classCount[vote] += 1
	sortedClassCount = sorted(classCount.iteritems(),
						key = operator.itemgetter(1), reverse = True)
	return sortedClassCount[0][0]

def createTree(dataSet,labels):
	classList = [example[-1] for example in dataSet]
	# 下类别完全相同则停止继续划分
	if classList.count(classList[0]) == len(classList):
		return classList[0]
	# 下遍历完所有特征时返回出现次数最多的,用完了所有特征仍不能划分仅含唯一类别的分组
	# 即无法简单返回唯一的类标签，因此使用上一个函数挑选出的最多次数类别返回
	if len(dataSet[0]) == 1:
		return majorityCnt(classList)
	bestFea = chooseBestFeatureToSplit(dataSet)
	bestFeaLabel = labels[bestFea]
	myTree = {bestFeaLabel:{}}
	# 下得到列表包含的所属属性值
	del(labels[bestFea])
	feaValues = [example[bestFea] for example in dataSet]
	uniqueVals = set(feaValues)
	for value in uniqueVals:
		# 下复制类标签，使得tree不能改变原来的列表
		subLabels = labels[:]
		# 下递归调用，返回值插入到字典变量myTree，函数终止时字典中将会嵌套很多
		# 代表叶子结点的字典数据
		myTree[bestFeaLabel][value] = createTree(splitDataSet\
									(dataSet, bestFea, value), subLabels)
	return myTree


# classify函数不同
# def classify(inputTree,featLabels,testVec):
    # firstStr = inputTree.keys()[0]
    # secondDict = inputTree[firstStr]
    # featIndex = featLabels.index(firstStr)
    # key = testVec[featIndex]
    # valueOfFeat = secondDict[key]
    # if isinstance(valueOfFeat, dict): 
        # classLabel = classify(valueOfFeat, featLabels, testVec)
    # else: classLabel = valueOfFeat
    # return classLabel
	

def classify(inputTree,feaLabels,testVec):
	firstStr = inputTree.keys()[0]
	secondDict = inputTree[firstStr]
	# 将标签字符串转换为索引
	feaIndex = feaLabels.index(firstStr)
	for key in secondDict.keys():
		if testVec[feaIndex] == key:
			if type(secondDict[key]).__name__ == 'dict':
				classLabel = classify(secondDict[key],feaLabels,testVec)
			else:
				classLabel = secondDict[key]
	return classLabel

def storeTree(inputTree, filename):
	import pickle
	fw = open(filename,'w')
	pickle.dump(inputTree,fw)
	fw.close()
	
def grabTree(filename):
	import pickle
	fr = open(filename)
	return pickle.load(fr)