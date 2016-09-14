# !/usr/bin/env python
from pyspark import SparkContext, SparkConf
import pickle

conf = SparkConf()
conf.setMaster("spark://dmlhdpc10:7077")
conf.setAppName("VFKMMProject")
conf.set("spark.executor.memory", "5g")
conf.set("spark.ui.port", "44041")
sc = SparkContext(conf=conf, pyFiles=['lib.zip'])
#numOfCores = 96
# sc = SparkContext(appName="PythonProject", pyFiles=['lib.zip'])
import numpy as np
import time
import argparse
from lib.splitter import split
from lib.bagger import get_size_no, partition, bag, pair, cartesian
from lib.kmm import computeBeta
from lib.evaluation import computeNMSE
from lib.scaleKMM import *
from lib.util import *
from lib.bagger import *
from lib.caculate import *
import csv


def kmmProcess():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', "--bagging", type=int, choices=[1, 2, 3, 4], default=1, help="bagging strategy")
    parser.add_argument("-t", "--training", type=int, default=12000, help="size of training data")
    # parser.add_argument("-s", "--tr_bsize", type=int, help="the sample size of train set")
    parser.add_argument("-x", "--te_bsize", type=int, help="the sample size of test set")
    # parser.add_argument("-m", "--train_samples", type=int, help="number of samples from training")
    parser.add_argument("-n", "--test_samples", type=int, help="number of samples from test")
    parser.add_argument("-e", "--eta", type=float, help="the eta value")
    parser.add_argument("-i", "--input", type=str, default='/home/wzyCode/scalablelearning/dataset/kdd.arff',
                        help="default input file")
    parser.add_argument("-o", "--operate", type=int, help="which experiment")
    parser.add_argument("-c", "--core", type=int, help="the number of cores")
    # parser.add_argument("-o", "--output", type=str, default='/home/wzyCode/scalablelearning/dataset/bag.txt',
    #                    help="default output file")
    args = parser.parse_args()

    mode = args.bagging  # bagging strategy
    training_size = args.training  # training set size (small training set)
    # tr_bsize = args.tr_bsize  # By default, the train bag size is dynamic, if specified, the train bag size will fix
    te_bsize = args.te_bsize  # By default, the test bag size is dynamic, if specified, the test bag size will fix
    # m = args.train_samples  # take m samples from training
    n = args.test_samples  # take n samples from
    eta = args.eta  # take eta value from
    o = args.operate
    numOfCores = args.core

    file_name = args.input;
    trainFileName = '/home/wzyCode/scalablelearning/input/' + file_name + '/' + str(training_size) + '_train.txt'  # input file path
    testFileName = '/home/wzyCode/scalablelearning/input/' + file_name + '/' + str(training_size) + '_test.txt'  # input file path
    betaFileName = '/home/wzyCode/scalablelearning/input/' + file_name + '/' + str(training_size) + '_beta.txt'  # input file path
    base_output_file = '/home/wzyCode/scalablelearning/output/' + str(o) + '/VFKMM_' + file_name + '_'

    # Step 1: Generate biased train and test set, as well as the orginal beta for train set
    #start = time.time()

    with open(trainFileName, 'rb') as f:
        train = pickle.load(f)
    with open(testFileName, 'rb') as f:
        test = pickle.load(f)
    with open(betaFileName, 'rb') as f:
        train_beta = pickle.load(f)


    #input_file = '/home/wzyCode/scalablelearning/dataset/' + file_name + '.arff'  # input file path    

    #train, train_beta, test = split(input_file, training_size)


    train_data = np.array(train)
    test_data = np.array(test)
    orig_beta_data = np.array(train_beta)
    # 1.Bagging process


    tr_bsize, m = get_train_info(train_data, n, eta)
    start = time.time()

    # Bagging the train and test data from the sampled index
    tr_bag_size, tr_bag_no = get_size_no(train_data, tr_bsize, m)
    te_bag_size, te_bag_no = get_size_no(test_data, te_bsize, n)
    print "tr_bag_size", tr_bag_size
    print "tr_bag_no", tr_bag_no
    print "te_bag_size", te_bag_size
    print "te_bag_no", te_bag_no

    bags = []

    if mode == 1:  # if test is too big, provide x or n to partition test set
        tr_n = bag(train_data, size=tr_bag_size, sample_no=tr_bag_no)
        te_n = partition(test_data, part_size=te_bag_size, part_no=te_bag_no)
    elif mode == 2:  # if train is too big, provide s or m to partition train set
        tr_n = partition(train_data, part_size=tr_bag_size, part_no=tr_bag_no)
        te_n = bag(test_data, size=te_bag_size, sample_no=te_bag_no)
    else:  # random sample, no partition
        tr_n = bag(train_data, size=tr_bag_size, sample_no=tr_bag_no)
        te_n = partition(test_data, part_size=te_bag_size, part_no=te_bag_no)

    # broadcast relative value:
    train_data_broad = sc.broadcast(train_data)
    test_data_broad = sc.broadcast(test_data)
    train_index = sc.broadcast(tr_n)
    test_index = sc.broadcast(te_n)

    print "tr_n", len(tr_n)
    print "te_n", len(te_n)

    if mode < 4:
        bags = cartesianVFKMM(tr_n, te_n)

    else:
        bags = pair(train_data, test_data, tr_n, te_n, sample_no=min(tr_bag_no, te_bag_no))

    numOfMaps = min(numOfCores, len(tr_n) * len(te_n))
    rdd = sc.parallelize(bags, numOfMaps)
    print("Number of splits: ", rdd.getNumPartitions())
    end = time.time()
    bagging_time = end - start

    # 2. Compute Beta Process
    # train_data = train_data_broad.value
    # test_data = test_data_broad.value

    start = time.time()
    res = rdd.map(
        lambda (idx, tr, te): computeBeta(idx, train_data_broad.value[tr], test_data_broad.value[te])).flatMap(
        lambda x: x)
    # res = rdd.map(lambda (idx, tr, te): computeBeta(idx, tr, te)).flatMap(lambda x: x)

    rdd1 = res.aggregateByKey((0, 0), lambda a, b: (a[0] + b, a[1] + 1),
                              lambda a, b: (a[0] + b[0], a[1] + b[1]))

    est_beta_map = rdd1.mapValues(lambda v: v[0] / v[1]).collectAsMap()
    est_beta_idx = est_beta_map.keys()

    end = time.time()
    compute_time = end - start

    # #

    # 3. Compute the NMSE between the est_beta and orig_beta through KMM
    start = time.time()

    est_beta = [est_beta_map[x] for x in est_beta_idx]
    orig_beta = orig_beta_data[est_beta_idx]
    final_result = computeNMSE(est_beta, orig_beta)

    end = time.time()
    evaluate_time = end - start

    # #4. statistics
    statistics = "In KMM method, mode=%s, train_size=%i, test_size=%i, size_of_train_samples=%i, number_of_train_samples=%i, size_of_test_samples=%i, K=%i,eta=%s\n" % \
                 (mode, len(train_data), len(test_data), tr_bag_size, tr_bag_no, te_bag_size, te_bag_no, eta)
    total_time = bagging_time + compute_time + evaluate_time
    time_info = "bagging_time=%s, compute_time=%s, evaluate_time=%s, total_time=%s\n" % \
                (bagging_time, compute_time, evaluate_time, total_time)
    print statistics
    print time_info

    message = "The final NMSE for KMM is : %s \n" % final_result
    print message

    print "---------------------------------------------------------------------------------------------"

    txt_output_file = base_output_file + '_K=' + str(n) + '_trainSize=' + str(training_size) + '_eta' + str(
        eta) + '_tr_bag_no' + str(tr_bag_no)

    ori_beta_val = []
    for i in est_beta_idx:
        ori_beta_val.append([i, orig_beta_data[i]])

    est_beta_val = []
    for i in est_beta_idx:
        est_beta_val.append([i, est_beta_map[i]])

    # write in text file
    textFile = txt_output_file + '.txt'
    print textFile
    with open(textFile, 'a') as textFile:

        textFile.write(statistics)
        textFile.write(time_info)
        textFile.write(message)

        textFile.write("The ori beta value is:")
        textFile.write('\n')
        textFile.write(str(ori_beta_val))

        textFile.write('\n')

        textFile.write("The est beta value is:")
        textFile.write('\n')
        textFile.write(str(est_beta_val))

    # write in csv file
    if o == 1:
        csvFile = base_output_file + '_K=' + str(n) + '_eta=' + str(eta) + '.csv'
        csvwrite = file(csvFile, 'a+')
        writer = csv.writer(csvwrite)
        if training_size == 100:
            writer.writerow(['train_size', 'accuracy', 'bagging_time', 'compute_time'])
            writer.writerow([training_size, final_result, bagging_time, compute_time])
        else:
            writer.writerow([training_size, final_result, bagging_time, compute_time])

    if o == 2:
        csvFile = base_output_file + '_trainSize=' + str(training_size) + '_eta=' + str(eta) + '.csv'
        csvwrite = file(csvFile, 'a+')
        writer = csv.writer(csvwrite)
        if n == 5:
            writer.writerow(['k_value', 'accuracy', 'bagging_time', 'compute_time'])
            writer.writerow([n, final_result, bagging_time, compute_time])
        else:
            writer.writerow([n, final_result, bagging_time, compute_time])

    if o == 3:
        csvFile = base_output_file + '_trainSize=' + str(training_size) + '_K=' + str(n) + '.csv'
        csvwrite = file(csvFile, 'a+')
        writer = csv.writer(csvwrite)
        if eta == 0.1:
            writer.writerow(['eta_value', 'accuracy', 'bagging_time', 'compute_time'])
            writer.writerow([eta, final_result, bagging_time, compute_time])
        else:
            writer.writerow([eta, final_result, bagging_time, compute_time])

    if o == 4:
        csvFile = base_output_file + '_trainSize=' + str(training_size) + '_K=' + str(n) + '_eta=' + str(eta) + '.csv'
        csvwrite = file(csvFile, 'a+')
        writer = csv.writer(csvwrite)
        if numOfCores == 20:
            writer.writerow(['numOfCores', 'accuracy', 'bagging_time', 'compute_time'])
            writer.writerow([numOfCores, final_result, bagging_time, compute_time])
        else:
            writer.writerow([numOfCores, final_result, bagging_time, compute_time])

        # csvFile = base_output_file + '.csv'
        # csvwrite = file(csvFile, 'a+')
        # writer = csv.writer(csvwrite)
        # writer.writerow([len(train_data), final_result, bagging_time, compute_time, total_time])


if __name__ == '__main__':
    kmmProcess()
    
