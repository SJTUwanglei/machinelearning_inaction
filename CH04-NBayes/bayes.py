# -*- coding: utf-8 -*-

from numpy import *				

def loadDataSet():
	postingList=[['my', 'dog', 'has', 'flea', \
					'problems', 'help', 'please'],
				 ['maybe', 'not', 'take', 'him', \
					'to', 'dog', 'park', 'stupid'],
				 ['my', 'dalmation', 'is', 'so', 'cute', \
					'I', 'love', 'him'],
				 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
				 ['mr', 'licks', 'ate', 'my', 'steak', 'how',\
					'to', 'stop', 'him'],
				 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
	classVec = [0,1,0,1,0,1]	# 1是侮辱性abusive,0不是
	return postingList,classVec

# 创建一个包含所有文档中出现的不重复词的列表，为此使用set	
def createVocabList(dataSet):
	# 创建空set
	vocabSet = set([])
	for document in dataSet:
		# 创建两个sets的单元
		vocabSet = vocabSet | set(document)		# | 用于求解集合并集
	return list(vocabSet)

# def setOfWords2Vec(vocabList, inputSet):
	# returnVec = [0] * len(vocabList)
	# for word in inputSet:
		# if word in vocabList:
			# returnVec[vocabList.index(word)] = 1
		# else:
			# print "the word: %s is not in my Vocabulary!" % word
	# return returnVec

# 上面是基于词集模型，即每个词出现与否，下面是基于词袋模型，即不止出现一次
def setOfWords2Vec(vocabList, inputSet):
	returnVec = [0] * len(vocabList)
	for word in vocabList:
		if word in vocabList:
			returnVec[vocabList.index(word)] += 1
	return returnVec

# 参数是文档矩阵trainMatrix，每篇文档类别标签所构成的向量	
def trainNB0(trainMatrix,trainCategory):
	numTrainDocs = len(trainMatrix)
	numWords = len(trainMatrix[0])
	pAbusive = sum(trainCategory)/float(numTrainDocs)
	p0Num = ones(numWords); p1Num = ones(numWords)
	p0Denom = 2.0; p1Denom = 2.0
	# p0Num = zeros(numWords); p1Num = zeros(numWords)
	# p0Denom = 0.0; p1Denom = 0.0
	for i in range(numTrainDocs):
		if trainCategory[i] == 1:
			# 下两行向量相加
			p1Num += trainMatrix[i]				# 向量元素是0，和1
			p1Denom += sum(trainMatrix[i])
		else:
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])
	# 对每个元素做除法
	p1Vec = log(p1Num/p1Denom)		# 这里的log不是引用math.log()，因为操作不了矩阵
	p0Vec = log(p0Num/p0Denom)
	# p1Vec = p1Num/p1Denom
	# p0Vec = p0Num/p0Denom
	return p0Vec,p1Vec,pAbusive
	
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
	p1 = sum(vec2Classify * p1Vec) + log(pClass1)
	p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
	if p1 > p0:
		return 1
	else:
		return 0

def testingNB():
	listOfPosts,listClasses = loadDataSet()
	myVocabList = createVocabList(listOfPosts)
	trainMat = []
	for postinDoc in listOfPosts:
		trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
	p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
	testEntry = ['love', 'my', 'dalmation']
	thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
	print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)
	testEntry = ['stupid', 'garbage']
	thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
	print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)

def textParse(bigString):
	import re
	listOfTokens = re.split(r'\W*', bigString)
	return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
	docList = []; classList = []; fullText = []
	for i in range(1,26):   # 是1-25
		wordList = textParse(open('email/spam/%d.txt' % i).read())
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(1)
		wordList = textParse(open('email/ham/%d.txt' % i).read())
		docList.append(wordList)
		fullText.append(wordList)
		classList.append(0)
	vocabList = createVocabList(docList)
	trainingSet = range(50); testSet = []
	# 随机构建训练集，共50封里选择10封test
	for i in range(10):
		randIndex = int(random.uniform(0,len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		# 添加到测试集，也从训练集中删除
		# 这种随机选择数据的一部分作为训练集，而剩余部分作为测试集的过程
		# 称为留存交叉验证（hold-out cross validation）。假定现在只完成
		# 了一次迭代，那么为了更精确地估计分类器的错误率，就应该进行多次迭
		# 代后求出平均错误率。
		del(trainingSet[randIndex])
	trainMat = []; trainClasses = []
	# 遍历训练集的所有文档，对每封邮件基于词汇表并使用setOfWords2Vec()函数
	# 来构建词向量。这些词在traindNB0()函数中用于计算分类所需的概率
	for docIndex in trainingSet:
		trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
		trainClasses.append(classList[docIndex])
	p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
	errorCount = 0
	# 下四行对测试集分类
	for docIndex in testSet:
		wordVector = setOfWords2Vec(vocabList,docList[docIndex])
		if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
			errorCount += 1
			print "classification error",docList[docIndex]
	print 'the error rate is: ',float(errorCount)/len(testSet)
	
	# ##错误率较高，0.3，0.4之类的##