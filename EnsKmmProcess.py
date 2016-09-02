#!/usr/bin/env python
from pyspark import SparkContext, SparkConf

conf = SparkConf()
conf.setMaster("spark://dmlhdpc10:7077")
conf.setAppName("EnsKMMProject")
conf.set("spark.executor.memory", "5g")
conf.set("spark.ui.port", "44041")
sc = SparkContext(conf=conf, pyFiles=['lib.zip'])
# sc = SparkContext(appName="PythonProject", pyFiles=['lib.zip'])
import numpy as np
import time
import argparse
from lib.splitter import split
from lib.bagger import *
from lib.kmm import computeBeta
from lib.evaluation import computeNMSE
from lib.scaleKMM import *;
from lib.util import *
import csv
import pickle

def ensKmmProcess():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--training", type=int, default=12000, help="size of training data")
    #parser.add_argument("-x", "--te_bsize", type=int, help="the sample size of test set")
    parser.add_argument("-n", "--test_samples", type=int, help="number of samples from test")
    parser.add_argument("-o", "--operate", type=int, help="which experiment")
    parser.add_argument("-c", "--core", type=int, help="the number of cores")
    parser.add_argument("-i", "--input", type=str, default='./dataset/powersupply.arff', help="default input file")
    # parser.add_argument("-o", "--output", type=str, default='/home/wzyCode/scalablelearning/nmseEnsKmm.txt', help="default output file")
    args = parser.parse_args()

    training_size = args.training
    # tr_bsize = args.tr_bsize # By default, the train bag size is dynamic, if specified, the train bag size will fix
    #te_bsize = 0  # By default, the test bag size is dynamic, if specified, the test bag size will fix
    # m = args.train_samples # take m samples from training
    o = args.operate
    n = args.test_samples  # take n samples from test
    #eta = args.eta  # take eta value from
    file_name = args.input;
    numOfCores = args.core

    #input_file = '/home/wzyCode/scalablelearning/dataset/' + file_name + '.arff'  # input file path
    base_output_file = '/home/wzyCode/scalablelearning/output/EnsKMM/'+str(o)+'/'+file_name

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

    # m, tr_bsize = get_train_info(train_data, n, eta)
    training_size = len(train_data)
    testDataLength = len(test_data)
    te_bag_size = testDataLength / n
    te_bsizeValue = sc.broadcast(te_bag_size)

    # Bagging the train and test data from the sampled index
    start = time.time()
    tr_bag_size_ens = len(train_data)
    tr_bag_no_ens = 1
    te_bag_size_ens, te_bag_no_ens = get_size_no(test_data, 0, n)

    tr_n_ens = partition(train_data, part_size=tr_bag_size_ens, part_no=tr_bag_no_ens)
    te_n_ens = partition(test_data, part_size=te_bag_size_ens, part_no=te_bag_no_ens)

    #set data as broad cast value
    train_data_broad = sc.broadcast(train_data)
    test_data_broad = sc.broadcast(test_data)

    #bags_ens = cartesian(train_data, test_data, tr_n_ens, te_n_ens)
    bags_ens = cartesianVFKMM(tr_n_ens, te_n_ens)

    numOfMaps = min(numOfCores, len(tr_n_ens) * len(te_n_ens))
    rddEns = sc.parallelize(bags_ens, numOfMaps)
    #rddEns = sc.parallelize(bags_ens, numOfCores)
    #print("Number of splits: ", rddEns.getNumPartitions())

    end = time.time()
    ens_bagging_time = end - start

    # 2. Compute Beta Process
    start = time.time()
    # rddEns = rddEns.map(lambda (idx, tr, te): (len(idx), len(tr), len(te)))
    # print "rddEns",rddEns.take(5)
    # print "te_bsizeValue",te_bsizeValue.value
    #rddEns = rddEns.map(lambda (idx, tr, te): getEnsKmmBeta(idx, tr, te, te_bsizeValue.value)).flatMap(lambda x: x)
    #rddEns = rddEns.map(lambda (idx, tr, te): getEnsKmmBeta(idx, train_data_broad.value[tr], test_data_broad.value[te], te_bsizeValue.value)).flatMap(lambda x: x)
    rddEns = rddEns.map(lambda (idx, tr, te): computeBeta(idx, train_data_broad.value[tr], test_data_broad.value[te])).flatMap(
    lambda x: x)


    rddEns = rddEns.aggregateByKey((0, 0), lambda a, b: (a[0] + b, a[1] + 1),
                                   lambda a, b: (a[0] + b[0], a[1] + b[1]))

    est_Ensbeta_map = rddEns.mapValues(lambda v: v[0] / v[1]).collectAsMap()
    est_Ensbeta_idx = est_Ensbeta_map.keys()
    end = time.time()
    compute_time_Ens = end - start

    # 3. Compute the NMSE between the est_beta and orig_beta through KMM
    start = time.time()

    est_Ensbeta = [est_Ensbeta_map[x] for x in est_Ensbeta_idx]
    orig_beta = orig_beta_data[est_Ensbeta_idx]
    final_result_Ens = computeNMSE(est_Ensbeta, orig_beta)

    end = time.time()
    evaluateEns_time = end - start

    # 4. statistics
    statisticsEns = "In EnsKMM method, train_size=%i, test_size=%i, tr_bag_size=%i, m=%i, te_bag_size=%i, n=%i\n" % \
                    (len(train_data), len(test_data), tr_bag_size_ens, tr_bag_no_ens, te_bag_size_ens, te_bag_no_ens)
    total_time = ens_bagging_time + compute_time_Ens + evaluateEns_time
    time_info_Ens = "bagging_time=%s, compute_time=%s, evaluate_time=%s, total_time=%s\n" % \
                    (ens_bagging_time, compute_time_Ens, evaluateEns_time, total_time)
    print statisticsEns
    print time_info_Ens

    messageEns = "The final NMSE for EnsKMM is : %s \n" % final_result_Ens
    print messageEns

    #write in txt file
    output_file = base_output_file + 'EnsKMM_K=' + str(n) + '_trainSize=' + str(training_size) + '.txt'

    ori_beta_val = []
    for i in est_Ensbeta_idx:
        ori_beta_val.append([i, orig_beta_data[i]])

    est_beta_val = []
    for i in est_Ensbeta_idx:
        est_beta_val.append([i, est_Ensbeta_map[i]])

    with open(output_file, 'a') as output_file:
        output_file.write(statisticsEns)
        output_file.write(time_info_Ens)
        output_file.write(messageEns)

        output_file.write("The ori beta value is:")
        output_file.write('\n')
        output_file.write(str(ori_beta_val))

        output_file.write('\n')

        output_file.write("The est beta value is:")
        output_file.write('\n')
        output_file.write(str(est_beta_val))


    #write in csv file
    if o == 1:
        csvFile = base_output_file + 'EnsKMM_K=' + str(n)  + '.csv'
        csvwrite = file(csvFile, 'a+')
        writer = csv.writer(csvwrite)
        if training_size == 100:
            writer.writerow(['train_size', 'accuracy', 'bagging_time', 'compute_time'])
            writer.writerow([training_size, final_result_Ens, ens_bagging_time, compute_time_Ens])
        else:
            writer.writerow([training_size, final_result_Ens, ens_bagging_time, compute_time_Ens])

    if o == 2:
        csvFile = base_output_file + 'EnsKMM_trainSize=' + str(training_size)  + '.csv'
        csvwrite = file(csvFile, 'a+')
        writer = csv.writer(csvwrite)
        if n == 5:
            writer.writerow(['k_value', 'accuracy', 'bagging_time', 'compute_time'])
            writer.writerow([training_size, final_result_Ens, ens_bagging_time, compute_time_Ens])
        else:
            writer.writerow([training_size, final_result_Ens, ens_bagging_time, compute_time_Ens])

if __name__ == '__main__':
    ensKmmProcess()