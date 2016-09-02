import math, numpy
import sys, random, time
from util import *
from manager import Manager
from util import computeKernelWidth
import numpy as np


#Compute beta
def cenKmm(traindata, testdata, gammab, maxFeature):
	#print 'Converting train sparse to array ...'
	#Xtrain = convertSparseToList(traindata, maxFeature)
	#print 'Converting test sparse to array ...'
	#Xtest = convertSparseToList(testdata, maxFeature)
	betai, runTime = kmm(traindata, testdata, gammab)
	return betai, runTime

def getCenKmmBeta(train, test):
	#maxFeature = len(train[0])
	maxFeature = train.shape[1]
	gammab = computeKernelWidth(train)
	res = cenKmm(train, test, gammab, maxFeature)
	beta = res[0]
	
	return beta

#Ensemble Beta - Dividing test data
def testEnsKmm(origtraindata, origtestdata, gammab, sampleSize, maxFeature):
	#Get sample
	ensBeta = []
	totalTime = 0

	testdata = list(origtestdata)
	#testdata = np.array(origtestdata) 
	
	#print 'Converting train sparse to array ...'
	#Xtrain = convertSparseToList(origtraindata, maxFeature)
	Xtrain = np.array(origtraindata) 
	
	print 'Running Test split KMMs ... '
	count = 0
	while len(testdata) > 0:
		newtestdata = []
		if len(testdata) <= (sampleSize):
			for j in xrange(len(testdata)):
				newtestdata.append(testdata[j])
			testdata = []
		else:
			for j in xrange(sampleSize):
				index = random.randint(0, len(testdata)-1)
				newtestdata.append(testdata[index])
				del testdata[index]

		#print 'Converting test split sparse to array ...'
		#Xtest = convertSparseToList(newtestdata, maxFeature)
		Xtest = newtestdata
		betai, runTime = kmm(Xtrain, Xtest, gammab)

		totalTime += runTime
		count += 1

		#combine beta (alpha * beta)
		alpha = float(len(newtestdata))/len(origtestdata)
		if len(ensBeta) == 0:
			wbeta = alpha * numpy.array(betai)
			ensBeta = list(wbeta.tolist())
		else:
			wbeta = alpha * numpy.array(betai)
			wbetaList = list(wbeta.tolist())
			for i in xrange(len(ensBeta)):
				ensBeta[i] += wbetaList[i]
	
	if count > 0:
		return ensBeta, float(totalTime)/count, totalTime
	else:
		return ensBeta, 0.0, totalTime


def getEnsKmmBeta(idx, train, test,sampleSize):
	#maxFeature = len(train[0])
	maxFeature = len(train[0])
	gammab = computeKernelWidth(np.array(train))
	result = testEnsKmm(train,test,gammab,sampleSize,maxFeature)
	beta = result[0]
	
	return [(x, y) for x, y in zip(idx, beta)]


	
#Ensemble Beta - Dividing train data
def trainEnsKmm(origtraindata, origtestdata, gammab, sampleSize, maxFeature):
	#Get sample
	ensBeta = []
	totalTime = 0

	for i in xrange(len(origtraindata)):
		ensBeta.append(0.0)

	traindata = list(origtraindata)

	print 'Converting test sparse to array ...'
	Xtest = convertSparseToList(origtestdata, maxFeature)

	print 'Running Train Split KMMs ... '
	bsum = 0.0
	count = 0
	while len(traindata) > 0:
		newtraindata = []
		newindex = []
		if len(traindata) <= (sampleSize):
			for j in xrange(len(traindata)):
				newtraindata.append(traindata[j])
			traindata = []
		else:
			for j in xrange(sampleSize):
				index = random.randint(0, len(traindata)-1)
				newtraindata.append(traindata[index])
				newindex.append(index)
				del traindata[index]

		print 'Converting train split sparse to array ...'
		Xtrain = convertSparseToList(newtraindata, maxFeature)
		betai, runTime = kmm(Xtrain, Xtest, gammab)

		totalTime += runTime
		count += 1

		#combine beta (beta)
		for i in xrange(len(newindex)):
			ensBeta[newindex[i]] = betai[i]
			bsum += betai[i]


	print 'Normalizing Beta ... '
	for b in ensBeta:
		b /= bsum
		b *= len(ensBeta)
	
	if count > 0:
		return ensBeta, float(totalTime)/count, totalTime
	else:
		return ensBeta, 0.0, totalTime


#Computing bagged train beta with replacement
def scaleKmm(traindata, testdata, gammab, sampleSize, numSample, maxFeature):
	dict = {}
	totalTime = 0
	bagBeta = []
	bagSampled = []



	for i in xrange(len(traindata)):
		dict[i] = []

	# Manager.logger.info('Bagging train - Number of samples : ' + str(numSample))

	print 'Converting test sparse to array ...'
	Xtest = convertSparseToList(testdata, maxFeature)

	#get beta for each train sample
	for i in xrange(numSample):
		# betai, newselect, time = getRandTrainBeta(traindata, Xtest, gammab, sampleSize, maxFeature)
		#Get sample
		newtraindata = []
		newselect = []
		while len(newtraindata) < sampleSize:
			index = random.randint(0, len(traindata)-1)
			newselect.append(index)
			newtraindata.append(traindata[index])

		print 'Converting train sparse to array ...'

		Xtrain = convertSparseToList(newtraindata, maxFeature)
		betai, time = kmm(Xtrain, Xtest, gammab)

		totalTime += time
		for j in xrange(len(newselect)):
			dict[newselect[j]].append(betai[j])

	count = 0
	sumb = 0.0
	for i in xrange(len(dict)):
		if len(dict[i]) > 0:
			b = float(sum(dict[i]))/len(dict[i])
			bagBeta.append(b)
			bagSampled.append(i)
			sumb += b
		else:
			count += 1

	# Manager.logger.info('Bagging train - Ignoring ' + str(count) + ' training instances.')
	
	if numSample > 0:
		return bagBeta, bagSampled, float(totalTime)/numSample, totalTime
	else:
		return bagBeta, bagSampled, 0.0, totalTime


#Combination of scaleKMM with train sample and test split
def scaleEnsKmm(traindata, origtestdata, gammab, sampleSize, numSample, maxFeature):
	dict = {}
	totalTime = 0
	bagBeta = []
	bagSampled = []

	for i in xrange(len(traindata)):
		dict[i] = []

	# Manager.logger.info('Bagging train Ensemble test - Number of samples : ' + str(numSample))


	print 'Generate test splits'
	testdata = list(origtestdata)
	newtestsplit = []

	while len(testdata) > 0:
		newtestdata = []
		if len(testdata) <= (sampleSize):
			for j in xrange(len(testdata)):
				newtestdata.append(testdata[j])
			testdata = []
		else:
			for j in xrange(sampleSize):
				index = random.randint(0, len(testdata)-1)
				newtestdata.append(testdata[index])
				del testdata[index]
		newtestsplit.append(newtestdata)

	print 'Generate training samples'
	count = 0
	#get beta for each train sample
	for i in xrange(numSample):
		#Get sample
		newtraindata = [] 
		newselect = []
		while len(newtraindata) < sampleSize:
			index = random.randint(0, len(traindata)-1)
			newselect.append(index)
			newtraindata.append(traindata[index])

		Xtrain = convertSparseToList(newtraindata, maxFeature)

		#For each test split, compute KMM for the training split
		print 'Compute beta for all test splits for sample ' + str(i)
		ensBeta = []
		for testsplit in newtestsplit:

			Xtest = convertSparseToList(testsplit, maxFeature)
			betai, time = kmm(Xtrain, Xtest, gammab)

			totalTime += time
			count += 1

			#combine beta (alpha * beta)
			alpha = float(len(testsplit))/len(origtestdata)
			if len(ensBeta) == 0:
				wbeta = alpha * numpy.array(betai)
				ensBeta = list(wbeta.tolist())
			else:
				wbeta = alpha * numpy.array(betai)
				wbetaList = list(wbeta.tolist())
				for b in xrange(len(ensBeta)):
					ensBeta[b] += wbetaList[b]

		for j in xrange(len(newselect)):
			dict[newselect[j]].append(ensBeta[j])

	scount = 0
	sumb = 0.0
	for i in xrange(len(dict)):
		if len(dict[i]) > 0:
			b = float(sum(dict[i]))/len(dict[i])
			bagBeta.append(b)
			bagSampled.append(i)
			sumb += b
		else:
			scount += 1

	# Manager.logger.info('Bagging train Ensemble test - Ignoring ' + str(scount) + ' training instances.')
	
	if count > 0:
		return bagBeta, bagSampled, float(totalTime)/count, totalTime
	else:
		return bagBeta, bagSampled, 0.0, totalTime




