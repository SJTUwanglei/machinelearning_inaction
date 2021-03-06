# -*- coding: utf-8 -*-

from numpy import *

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1   # -1
    dataMat = []; labelMat = []
    fr = open(fileName) # 要重新开
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))   # txt里x0 = 1.0已经加进去了
        dataMat.append(lineArr)
        labelMat.append(lineArr[-1])
    return dataMat,labelMat

def standRegres(xArr,yArr):
    xMat = mat(xArr); yMat = mat(yArr).T    # .T, .transpose()
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:  # linalg是numpy的库，里面是线代函数，行列式 linalg.det()
        print "This matrix is singular, cannot do inverse"
        return
    ws = xTx.I * (xMat.T * yMat)    # 矩阵式表达, .I是逆
    return ws
# numpy的linear algebra library的另一个解矩阵 ws = linalg.solve(xTx,xMat.T * yMat)

def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xArr)[0]
    weights = mat(eye(m))
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = exp(diffMat * diffMat.T / (-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0:
        print "This matrix is singular and cannot be inverse"
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

def lwlrTestPlot(xArr,yArr,k=1.0):
    yHat = zeros(shape(yArr))
    xCopy = mat(xArr)
    xCopy.sort(0)           # sort for Plot
    for i in range(shape(xArr)[0]):
        yHat[i] = lwlr(xCopy[i],xArr,yArr,k)
    return yHat,xCopy

def rssError(yArr,yHatArr):
    return ((yArr - yHatArr)**2).sum()

def ridgeRegres(xMat,yMat,lam=0.2):
    xTx = xMat.T * xMat
    denon = xTx + eye(shape(xMat)[1])*lam
    if linalg.det(denon) == 0:
        print "the matrix is singular and cannot be inverse"
        return
    ws = denon.I * (xMat.T*yMat)
    return ws

def ridgeTest(xArr,yArr):   # 岭回归
    xMat = mat(xArr); yMat = mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean
    xMeans = mean(xMat,0)
    xVar = var(xMat,0)
    xMat = (xMat - xMeans)/xVar
    numTestPts = 30
    wMat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:] = ws.T
    return wMat

def regularize(xMat):   # 0均值，单位方差,数据标准化
    inMat = xMat.copy()
    inMeans = mean(inMat,0)
    inVar = var(inMat,0)
    inMat = (inMat - inMeans)/inVar
    return inMat

def stageWise(xArr,yArr,eps=0.01,numIt=100):    # 前向逐步线性回归
    xMat = mat(xArr); yMat = mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    m,n = shape(xMat)
    returnMat = zeros((numIt,n))
    ws = zeros((n,1)); wsTest = ws.copy(); wsMax = ws.copy()
    for i in range(numIt):
        print ws.T
        lowestError = inf
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A,yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:] = ws.T
    return returnMat


# 使用google API来预估乐高价格
from time import sleep
import json
import urllib2
def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    sleep(10)
    myAPIstr = '??????' # here you need a API
    searchURL = '......' % (myAPIstr, setNum)   # here you need a website
    pg = urllib2.urlopen(searchURL)
    retDict = json.loads(pg.read())
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else: newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if sellingPrice > origPrc * 0.5:
                    print "%d\t%d\t%d\t%f" % (yr,numPce,newFlag,origPrc, sellingPrice)
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except: print 'problem with item %d' % i

        
def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)
    
def crossValidation(xArr, yArr, numVal=10):
    m = len(yArr)
    indexList = range(m)
    errorMat = zeros((numVal,30))   # errorMat: numVal * 30
    for i in range(numVal):
        trainX = []; trainY = []
        testX = []; testY = []
        random.shuffle(indexList)   # shuffle()混洗函数，可实现训练集或测试集数据点随机选取
        for j in range(m):
            if j < m*0.9:   # first 90% train, last 10% test
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX, trainY)    # get 30 weight vectors from ridge 30个不同lam回归系数
        for k in range(30): # loop over all of the ridge estimates
            matTestX = mat(testX); matTrainX = mat(trainX)
            meanTrain = mean(matTrainX,0)
            varTrain = var(matTrainX,0)
            matTestX = (matTestX - meanTrain)/varTrain  # regularize test
            yEst = matTestX * mat(wMat[k,:]).T + mean(trainY)   
            errorMat[i,k] = rssError(yEst.T.A, array(testY))
            # print errorMat[i,k]
    meanErrors = mean(errorMat,0)
    minMean = float(min(meanErrors))
    bestWeights = wMat[nonzero(meanErrors==minMean)]
    
    xMat = mat(xArr); yMat = mat(yArr).T
    meanX = mean(xMat,0); varX = var(xMat,0)
    unReg = besWeights/varX
    print "the best model from Ridge regression si:\n",unReg
    print "with constant term: ",-1*sum(multiply(meanX,unReg))+mean(yMat)
    
    
    #  挖掘一下哪些特征重要哪些不重要