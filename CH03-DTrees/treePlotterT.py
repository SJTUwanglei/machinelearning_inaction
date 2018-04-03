# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

# 下面字典也可以写作 decisionNode = dict(boxstyle:'sawtooth',fc:'0.8')
# boxstyle为文本框类型，sawtooth是锯齿形，fc是边框线粗细
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
	# annotate是一个关于数据点的文本
	# nodeTxt为要显示的文本，centerPt为文本的中心点，箭头所在点，parentPt为指向点
	createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
		xytext=centerPt, textcoords='axes fraction', 
		va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)

def createPlot():
	fig = plt.figure(1, facecolor='white')	# 定义一个画布，背景白色
	fig.clf()	# 清空画布
	# createPlot.ax1是全局变量，绘制图像的句柄，subplot为定义了一个绘图
	# 111表示1行1列，第1个图
	# frameon表示是否绘制坐标轴矩形
	createPlot.ax1 = plt.subplot(111, frameon=False)
	plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
	plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
	plt.show()

def getNumLeafs(myTree):
	numLeafs = 0
	firstStr = myTree.keys()[0]
	secondDict = myTree[firstStr]
	for key in secondDict.keys():
		if type(secondDict[key]).__name__=='dict':	# 双下划线开头双下划线结尾name
			numLeafs += getNumLeafs(secondDict[key])
		else: numLeafs += 1
	return numLeafs

def getTreeDepth(myTree):
	maxDepth = 0
	firstStr = myTree.keys()[0]
	secondDict = myTree[firstStr]
	for key in secondDict.keys():
		if type(secondDict[key]).__name__=='dict':
			thisDepth = 1 + getTreeDepth(secondDict[key])
		else: thisDepth = 1
		if thisDepth > maxDepth: maxDepth = thisDepth
	return maxDepth

def retrieveTree(i):
	listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': \
					{0: 'no', 1: 'yes'}}}},
				   {'no surfacing': {0: 'no', 1: {'flippers': \
					{0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}]
	return listOfTrees[i]