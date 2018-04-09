# -*- coding: utf-8 -*- 

from numpy import *

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')    # 数据是x1，x2，类别0或1
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])]) # 将x0设为0
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn) # 获得输入数据并转化成numpy矩阵100*3
    labelMat = mat(classLabels).transpose() # classLabels开始是行向量,矩阵才能转置
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))
    for k in range(maxCycles):
        # 下三行矩阵乘法
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error  # (m*n).T * m*1
    return weights

def plotBestFit(weights):
    import matplotlib.pyplot as plt
    # weights = wei.getA()
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n =shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x) / weights[2]
    ax.plot(x,y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()
    
# 在cmd里要先定义 
# dataArr,labelMat = logRegres.loadDataSet()
# weights = logRegres.gradAscent(dataArr,labelMat)
# 定义晚后引用    logRegres.plotBestFit(weights.getA())
# 相应的在plot函数里， 去掉了  weights = wei.getA(),wei直接为weights
# 按照原书原代码会显示为 'numpy.ndarray' object has no attribute 'getA'

def stoGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)   # 这里是1*n,ones((n,1))是n*1
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights)) # 这里是1*n * 1*n,array乘法要sum
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]   # error是数
    return weights

def stoGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):    # 默认迭代次数可改
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01    # 变步长
    # 如果要处理的问题是动态变化的，那么可以适当加大
    # 上述常数项，来确保新的值获得更大的回归系
    # 数。另一点值得注意的是，在降低alpha的函数
    # 中，alpha每次减少1/(j+i) ，其中   %%j是迭代次
    # 数，    i是样本点的下标1 。%%       这样当j<<max(i)时，
    # alpha就不是严格下降的。避免参数的严格下降也
    # 常见于模拟退火算法等其他优化算法中。
            randIndex = int(random.uniform(0,len(dataIndex)))   # randomIndex
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex] 
            del(dataIndex[randIndex])
    return weights              # n*1, 如何保证全部经历，全部用完

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stoGradAscent1(array(trainingSet), trainingLabels, 500)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print "the error rate of test is: %f" % errorRate
    return errorRate

def multiTest():
    numTests = 10; errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print "after %d iterations the average error rate is:\
         %f" % (numTests, errorSum/float(numTests))