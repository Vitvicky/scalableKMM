def computeNMSE(estBeta, origBeta):
    estBetaSum = sum(estBeta)
    origBetaSum = sum(origBeta)


    nmse = 0
    for i in xrange(len(estBeta)):
        nmse += ((estBeta[i]/estBetaSum) - (origBeta[i]/origBetaSum)) ** 2

    return nmse / len(estBeta)
