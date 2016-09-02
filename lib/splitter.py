from scipy.io import arff
import math, numpy, sklearn.metrics.pairwise as sk
import logging, random

# create logger with 'spam_application'
logger = logging.getLogger('spam_application')


def computeMean(data):
    mean = {}
    for d in data:
        for (k, v) in enumerate(d):
            if mean.has_key(k):
                mean[k] += v
            else:
                mean[k] = v
    for (k, v) in mean.iteritems():
        mean[k] = v / len(data)
    print mean
    return mean


# Compute distance
def computeDistanceSq(d1, d2):
    dist = 0
    for i, e in enumerate(d1):
            dist += (e - d2[i]) ** 2
    return dist

# Compute standard deviation from mean of sparse data
def computeSTD(data, mean):
    print 'computing distance for STD'
    dist_list = []
    for d in data:
        dist = computeDistanceSq(d, mean)
        dist_list.append(dist)

    print 'Computing STD'
    return numpy.std(dist_list)


#Compute probability of sampling instance as training data
def computeProb(instance, mean, std):
    dist = computeDistanceSq(instance, mean)
    return math.exp(-1 * dist / std)


#Sample training data from input sparse data
def generateTrain(origdata, trainsize):
    train = []
    trainBeta = []
    data = list(origdata)
    print 'The total size is: %i' % len(data)

    logger.info('Computing mean & std')
    mean = computeMean(data)
    std = computeSTD(data, mean)

    logger.info('Generating training data')

    count = 0
    stdcount = 1
    origstd = std

    while len(train) < trainsize:
        d = random.randint(0, len(data)-1)
        p = computeProb(data[d], mean, std)
        x = random.uniform(0,1)
        if x < p:
            train.append(data[d])
            trainBeta.append(1.0/p)
            del data[d]
            count = 0
        else:
            count += 1

        if count == trainsize:
            count = 0
            stdcount += 1
            std = origstd*stdcount

    return train, trainBeta, data


def split(filename, train_size, reverse=False):
    data, meta = arff.loadarff(filename)
    orig_data = []
    for line in data:
        orig_data.append(list(line)[0:-1])
    if reverse:
        train_size = len(orig_data) - train_size
    return generateTrain(tuple(orig_data), train_size)


def main():
    filename = '../dataset/powersupply.arff'
    train, train_beta, data = split(filename, 2)
    print train
    print train_beta
    print len(data)

if __name__ == '__main__':
    main()
