# -*- coding: utf-8 -*-

'''
    using kNN on results from a dating site
    created on Jan,8th,2018
    author: wanglei
'''

from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt


# 导入数据
def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())     # know lines
    returnMat = zeros((numberOfLines,3))    # create a Numpy matrix
    classLabelVector = []
    fr = open(filename) #!!!!!!!!!!不能漏掉
    index = 0
    for line in fr.readlines():
        line = line.strip() 
        listFromLine = line.split('\t')     # return list
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))  # 这里注意转化问题,
                                                    # 不同类型的txt里面是不同的
                                                    # 这里用数字式，但是要转为int
        index += 1
    return returnMat,classLabelVector

# 归一化，等权重
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))     # 建立与dataSet结构一样的矩阵
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))    # new = (old - min)/(max - min)
    
    # for i in range(1,m):
        # normDataSet[i,:] = (dataSet[i,:] - minVals) / ranges
    
    return normDataSet,ranges,minVals

# kNN算法
def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]      #shape[0],first dimension length
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1) #first dimension distances
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()    #argsort(),从小到大排序，返回下标
    classCount = {}
    for i in range(k):                          # 这里运行后发现错误，下indicies少个字母i
        votelabel = labels[sortedDistIndicies[i]]   # 对k个样本的所属类别进行统计
        classCount[votelabel] = classCount.get(votelabel,0) + 1 #get()
    sortedClassCount = sorted(classCount.iteritems(),   #iteritems
     key = operator.itemgetter(1),reverse = True)
    return sortedClassCount[0][0]
    

# 测试，比例自定如10%
def datingClassTest():
    hoRatio = 0.10      #hold out 10%
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))
    print errorCount
    
# def datingClassTest():
    # hoRatio = 0.10
    # datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    # normMat, ranges, minVals = autoNorm(datingDataMat)
    # m = normMat.shape[0]
    # numTestVecs = int(m*hoRatio)      # 最后语法错误是因为这里多打了个s
    # errorCount = 0.0              # 虽然没报错，但是还发现这里r错打成e
    # for i in range(numTestVecs):
        # classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        # print "The classifier came back with: %d, the real answer \
                        # is: %d"% (classfierResult,datingLabels[i])    # 最后错误！转行符错用
        # if (classifierResult != datingLabels[i]): errorCount += 1.0
    # print "The total error rate is: %f" % (errorCount/float(numTestVecs)