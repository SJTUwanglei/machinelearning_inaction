# -*- coding: utf-8 -*-

# 在几百个点组成的小规模数据集上，简化版SMO算法的运行是没有什么问题的，更大不行

# SMO的算法，Miscrosoft Research的Platt，1998年论文里首次出现，最快的二次规划优化算法
# 关于SMO最好的资料就是他本人写的《Sequential Minimal Optimization A 
#       Fast Algorithm for Training Support Vector Machines》

from numpy import *

def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

def selectJrand(i,m):
    j = i
    while (j == i):
        j = int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):  # 无空格
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):   # 有空格
    dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose()    # 转置
    b = 0; m,n = shape(dataMatrix)
    alphas = mat(zeros((m,1)))  # alpha初始可行解选择的是[000000...]
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas,labelMat).T *\
                    (dataMatrix*dataMatrix[i,:].T)) + b
            Ei = fXi - float(labelMat[i])
            # 如果alpha可以更改进入优化过程，且检查alpha不能等于0或C,已经在边界上不能再增大或减小，不值得再优化
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or \
                   ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i,m)
                fXj = float(multiply(alphas,labelMat).T*\
                        (dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy();
                alphaJold = alphas[j].copy();
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] -alphas[i])
                    H = min(C, C + alphas[j] -alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H: print "L==H"; continue
                eta = 2.0 * dataMatrix[i,:] * dataMatrix[j,:].T - \
                    dataMatrix[i,:] * dataMatrix[i,:].T - \
                    dataMatrix[j,:] * dataMatrix[j,:].T
                if eta >= 0: print "eta >= 0"; continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta  # alpha的new uncut
                alphas[j] = clipAlpha(alphas[j],H,L)        # alpha的new,直接更新
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print "j not moving enough";continue
                alphas[i] += labelMat[j] * labelMat[i] *\
                            (alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) *\
                            dataMatrix[i,:]*dataMatrix[i,:].T - \
                            labelMat[j] * (alphas[j] - alphaJold) *\
                            dataMatrix[i,:] * dataMatrix[j,:].T
                b2 = b - Ej - labelMat[i] * (alphas[i] -alphaIold) *\
                            dataMatrix[i,:] * dataMatrix[j,:].T - \
                            labelMat[j] * (alphas[j] - alphaJold) *\
                            dataMatrix[j,:] * dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print "iter: %d  i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
        if (alphaPairsChanged == 0): iter += 1
        else: iter = 0
        print "iteration number: %d" % iter
    return b,alphas
   
# 建立一个数据结构保存所有重要值，构建一个仅包含init方法的optStruct类

# class optStruct:
    # def __init__(self, dataMatIn, classLabels, C, toler):
        # self.X = dataMatIn
        # self.labelMat = classLabels
        # self.C = C
        # self.tol = toler
        # self.m = shape(dataMatIn)[0]
        # self.alphas = mat(zeros((self.m,1)))
        # self.b = 0
        # # 误差缓存
        # self.eCache = mat(zeros((self.m,2)))

def kernelTrans(X, A, kTup):
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0] =='lin': K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow * deltaRow.T
        # numpy矩阵中，除法意味着对矩阵元素展开计算而不像在matlab中计算矩阵的逆。
        K = exp(K /(-1*kTup[1]**2))
    # 如果遇到无法识别的元组，程序就会抛出异常，此时不希望程序再运行。
    else: raise NameError('Husto We Have a Problem -- That Kernel is not recognized')
    return K

# 加了核函数的数据结构
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2)))
        self.K = mat(zeros((self.m,self.m)))
        # 全局的K值只需计算一次。然后，当想要使用核函数时，就可以对它进行调用
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)

def calcEk(oS, k):
    # fXk = float(multiply(oS.alphas,oS.labelMat).T *\
            # (oS.X * oS.X[k,:].T)) + oS.b
    fXk = float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k]) + oS.b # 2018.02.27 把括号放进里面
    Ek = fXk - float(oS.labelMat[k])
    return Ek
    
def selectJ(i, oS, Ei): # 参数oS
    # 内循环的启发式方法
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1,Ei]
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]  # nonezero返回的是非零E值下
                                                    # 对应的alpha值。
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i: continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            # 以下两行选择具有最大步长的j，用于选择合适第二个alpha以保证最大步
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej
    
def updateEk(oS, k):
    Ek = calcEk(oS, k)  # 计算误差值并存入缓存，对alpha优化后会用到这个值
    oS.eCache[k] = [1,Ek]

def innerL(i, oS):  # 使用自己的数据结构，该结构在参数oS中传递
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or\
        ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        # 第二个alpha选择中的启发式方法，用的是selecJ而不是selectJrand
        j,Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: print "L==H"; return 0 # 注意，这里就return 0了，与上面continue不同
        eta = 2.0 * oS.K[i,j] -oS.K[i,i] - oS.K[j,j]
        # eta = 2.0 * oS.X[i,:] * oS.X[j,:].T - oS.X[i,:] * oS.X[i,:].T - \
                    # oS.X[j,:] * oS.X[j,:].T
        if eta >= 0: print "eta >= 0"; return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta    # Ej和Ei的差位置换了
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        # 更新错误差值缓存，在alpha更新时Ecache更新
        updateEk(oS,j)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print "j not moving enough"; return 0    
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] *\
                            (alphaJold - oS.alphas[j])
        # 更新错误差值缓存
        updateEk(oS, i)
        # ******!!!!!!更新完Ei-Ej后，程序进入了not moving enough 下面的循环
        # ******!!!!!!!!!!唯二的问题另一是alphaJold写成了oS.alphaJold
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold)*oS.K[i,j] -\
                    oS.labelMat[j]*(oS.alphas[j] - alphaJold)*oS.K[i,j]
        # b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) *\
                    # oS.X[i,:]*oS.X[i,:].T - oS.labelMat[j] *\
                    # (oS.alphas[j] - alphaJold) * oS.X[i,:]*oS.X[j,:].T
        # 我tm写完了完整的Platt SMO，跑了一下没问题，用kernel时才发现b2的式子没写
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold)*oS.K[i,j] -\
                    oS.labelMat[j]*(oS.alphas[j] - alphaJold)*oS.K[j,j]
        # b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold) *\
                    # oS.X[i,:]*oS.X[j,:].T - oS.labelMat[j] *\
                    # (oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2) / 2.0
        return 1
    else: return 0

# 选择第一个alpha的外循环
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler,kTup)    # smoP也要改
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print "fullSet, iter: %d i:%d, pairs changed %d" %(iter,i,alphaPairsChanged)
            iter += 1       # 上面这个print要缩进吗,事实上对比下面的这个是要缩进的
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print "non-bound, iter: %d i:%d, pairs changed %d" %\
                        (iter,i,alphaPairsChanged)
            iter += 1
        if entireSet: entireSet = False
        elif (alphaPairsChanged == 0): entireSet = True
        print "iteration number: %d" % iter
    return oS.b,oS.alphas

# 当迭代次数超过指定的最大值，或者遍历整个集合都未对任意alpha对进行修改
# 时，就退出循环.这里的maxIter变量和函数smoSimple()中的作用有一
# 点不同，后者当没有任何alpha发生改变时会将整个集合的一次遍历过程计成一次迭代，
# 而这里的一次迭代定义为一次循环过程，而不管该循环具体做了什么事.

# while循环的内部与smoSimple()中有所不同，一开始的for循环在数据集上遍历任意可能的
# alpha。我们通过调用innerL()来选择第二个alpha，并在可能时对其进行优化处理。如果有任
# 意一对alpha值发生改变，那么会返回 1。第二个for循环遍历所有的非边界alpha值，也就是不
# 在边界0或C上的值。接下来，我们对for循环在非边界循环和完整遍历之间进行切换，并打
# 印出迭代次数。最后程序将会返回常数b和alpha值

# 常数C一方面要保障所有样例的间隔不小于1.0，另一方面又要使得分类间隔要尽可能大，
# 并且要在这两方面之间平衡。如果C很大，那么分类器将力图通过分隔超平面对所有的
# 样例都正确分类

def calcWs(alphas,dataArr,classLabels):
    X = mat(dataArr); labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

# 调参数，径向基核函数参数sigma，即k1的影响
def testRbf(k1=1.3):
    dataArr,labelArr = loadDataSet('testSetRBF.txt')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
    datMat = mat(dataArr); labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A>0)[0]
    # 创建支持向量
    sVs = datMat[svInd]
    labelSV = labelMat[svInd];
    print "there are %d Spport Vectors" % shape(sVs)[0]
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        # 给出了如何利用核函数分类，只需要支持向量就可以分类
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict = kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print "the training error rate is: %f" % (float(errorCount/m))
    dataArr, labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    datMat = mat(dataArr); labelMat = mat(labelArr).transpose() # 注意！是datMat
    m,n = shape(datMat) 
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict = kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print "the test error rate is: %f" % (float(errorCount)/m)

# training error rete is; 0.000000     test error is: 0.52000000
# 改了上面一个dataMat为datMat后，分别 0.000000，   0.410000
# 在运行一次， 0.000000，   0.150000

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9: hwLabels.append(-1)    # 碰到数字9输出类别标签-1，因为
                                                    # 数据就是1和9，代表两类
        else: hwLabels.append(1)
        trainingMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels

# SVM做多类分类器见论文 A Comparison of Methods for Multiclass
# Support Vector Machines

def testDigits(kTup=('rbf', 10)):
# 函数元组kTup是输入参数，而testRbf()中默认的就是使用rbf核函数。若不增加任何参数
# 输入，则默认kTup = ('rbf',10),如果输入则可以是svmMliA.testDigits(('rbf', 20))
    dataArr, labelArr = loadImages('trainingDigits')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat = mat(dataArr); labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    print "there are %d Support Vectors" % shape(sVs)[0]
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict = kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print "the training error rate is: %f" % (float(errorCount)/m)
    dataArr, labelArr = loadImages('testDigits')
    errorCount = 0
    datMat = mat(dataArr); labelmat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict = kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print "the test error rate is: %f" % (float(errorCount)/m)

# 第一次（'rbf',20) training error: 0.000000   test error; 0.021505   56个SV
# 第二次（'rbf',20) 0.000000    0.010753     16次迭代     50个SV
# ('rbf',10) 0.000000    0.005376      20次迭代， 127个SV 是最低的