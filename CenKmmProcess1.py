#!/usr/bin/env python
# from pyspark import SparkContext
import argparse
import numpy as np
import time
from lib.kmm import computeBeta
from lib.evaluation import computeNMSE
from lib.scaleKMM import *;
from lib.util import *
import pickle
import csv
from lib.splitter import split

def cenKmmProcess():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default='./dataset/powersupply.arff', help="default input file")
    parser.add_argument("-t", "--training", type=int, default=12000, help="size of training data")
    # parser.add_argument("-o", "--output", type=str, default='/home/wzyCode/scalablelearning/nmseCenKMM.txt', help="default output file")
    args = parser.parse_args()
    file_name = args.input;
    training_size = args.training
    
    #input_file = '/home/wzyCode/scalablelearning/dataset/' + file_name + '.arff'  # input file path
    trainFileName = '/home/wzyCode/scalablelearning/input/' + file_name + '/' + str(training_size) + '_train.txt'  # input file path
    testFileName = '/home/wzyCode/scalablelearning/input/' + file_name + '/' + str(training_size) + '_test.txt'  # input file path
    betaFileName = '/home/wzyCode/scalablelearning/input/' + file_name + '/' + str(training_size) + '_beta.txt'  # input file path

    with open(trainFileName, 'rb') as f:
        train = pickle.load(f)
    with open(testFileName, 'rb') as f:
        test = pickle.load(f)
    with open(betaFileName, 'rb') as f:
        train_beta = pickle.load(f)
    
    #train, train_beta, test = split(input_file, training_size)    

    train_data = np.array(train)
    test_data = np.array(test)
    orig_beta_data = np.array(train_beta)

    # Step 1: Compute the estimated beta from cenKMM
    start = time.time()
    maxFeature = train_data.shape[1]
    gammab = computeKernelWidth(train_data)
    res = cenKmm(train_data, test_data, gammab, maxFeature)
    est_Cenbeta = res[0]

    end = time.time()
    compute_time_Cen = end - start

    # Step 2: Compute the NMSE between the est_beta and orig_beta through CenKMM
    start = time.time()
    final_result_Cen = computeNMSE(est_Cenbeta, orig_beta_data)
    end = time.time()
    evaluateCen_time = end - start

    # Step 3: statistics
    statisticsCen = "In CenKMM method, train_size=%i, test_size=%i" % \
                    (len(train_data), len(test_data))
    total_time = compute_time_Cen + evaluateCen_time
    time_info_Cen = "compute_time=%s, evaluate_time=%s, total_time=%s\n" % \
                    (compute_time_Cen, evaluateCen_time, total_time)
    print statisticsCen
    print time_info_Cen

    messageCen = "The final NMSE for CenKMM is : %s \n" % final_result_Cen
    print messageCen

    print "---------------------------------------------------------------------------------------------"

    output_file = '/home/wzyCode/scalablelearning/output/CenKMM/CenKMM_' + file_name + '_trainSize' + str(
        training_size) + '.txt'

    with open(output_file, 'a') as output_file:
        output_file.write(statisticsCen)
        output_file.write(time_info_Cen)
        output_file.write(messageCen)

    #write in csv file
    csvFile = '/home/wzyCode/scalablelearning/output/CenKMM/CenKMM_' + file_name + '.csv'
    csvwrite = file(csvFile, 'a+')
    writer = csv.writer(csvwrite)
    if training_size == 100:
        writer.writerow(['train_size', 'accuracy', 'compute_time'])
        writer.writerow([training_size, final_result_Cen, compute_time_Cen])
    else:
        writer.writerow([training_size, final_result_Cen, compute_time_Cen])

if __name__ == '__main__':
    cenKmmProcess()