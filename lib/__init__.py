from scaleKMM import cenKmm
from scaleKMM import testEnsKmm
from util import *
from splitter import split
import numpy as np
from scipy.io import arff
from scaleKMM import scaleEnsKmm
from evaluation import computeNMSE

def maintest():
    path = '../dataset/powersupply.arff'
    #data, meta = arff.loadarff(path)
    #data, label, maxFeature = getArffData(path,1000)
    train, train_beta, test = split('../dataset/powersupply.arff', 300)
    #train, trainBeta, test = generateTrain(data, 300)
    train_data = np.array(train)
    test_data = np.array(test)
    maxFeature = train_data.shape[1]
    #maxFeature = len(train[0])
    print "maxFeature:" , maxFeature
    gammab = computeKernelWidth(train_data)
    print "gammab", gammab
    
    #print 'Converting train sparse to array ...'
    #Xtrain = convertSparseToList(train, maxFeature)
    #print 'Converting test sparse to array ...'
    #Xtest = convertSparseToList(test, maxFeature)
    beta,runtime = cenKmm(train_data, test_data, gammab, maxFeature)
    print beta
    
def main():
    path = '../dataset/powersupply.arff'
    #data, meta = arff.loadarff(path)
    #data, label, maxFeature = getArffData(path,1000)
    #train, trainBeta, test = generateTrain(data, 100)
    train, train_beta, test = split('../dataset/powersupply.arff', 500)
    maxFeature = len(train[0])
    sampleSize = len(train)
    #print "sampleSize:" , sampleSize
    gammab = computeKernelWidth(np.array(train))
    result = testEnsKmm(train,test,gammab,10,maxFeature)
    beta = result[0]
    print "beta",beta
    
if __name__ == '__main__':
    #sc = SparkContext(appName="PythonPartitioning", pyFiles=['lib.zip'])
    main()