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
    

    #input_file = '/home/wzyCode/scalablelearning/dataset/' + file_name + '.arff'  # input file path    
    with open(trainFileName, 'rb') as f:
        train = pickle.load(f)
    with open(testFileName, 'rb') as f:
        test = pickle.load(f)
    with open(betaFileName, 'rb') as f:
        train_beta = pickle.load(f)
    #train, train_beta, test = split(input_file, training_size)


    train_data = np.array(train)
    print "train_data",train_data
    test_data = np.array(test)
    orig_beta_data = np.array(train_beta)
    # 1.Bagging process


    tr_bsize, m = get_train_info(train_data, n, eta)
    start = time.time()

    # Bagging the train and test data from the sampled index
    tr_bag_size, tr_bag_no = get_size_no(train_data, tr_bsize, m)
    te_bag_size, te_bag_no = get_size_no(test_data, te_bsize, n)
    #print "tr_bag_size", tr_bag_size
    #print "tr_bag_no", tr_bag_no
    #print "te_bag_size", te_bag_size
    #print "te_bag_no", te_bag_no

    bags = []

    if mode == 1:  # if test is too big, provide x or n to partition test set
        tr_n = bag(train_data, size=tr_bag_size, sample_no=tr_bag_no)
        te_n = partition(test_data, part_size=te_bag_size, part_no=te_bag_no)
    elif mode == 2:  # if train is too big, provide s or m to partition train set
        tr_n = partition(train_data, part_size=tr_bag_size, part_no=tr_bag_no)
        te_n = bag(test_data, size=te_bag_size, sample_no=te_bag_no)
    else:  # random sample, no partition
        tr_n = bag(train_data, size=tr_bag_size, sample_no=tr_bag_no)
	print "bag",tr_n
        te_n = partition(test_data, part_size=te_bag_size, part_no=te_bag_no)

    


if __name__ == '__main__':
    kmmProcess()