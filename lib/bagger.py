import numpy as np
import math


def get_size_no(data, bag_size, sample_no):
    if bag_size:
        if sample_no:
            return bag_size, sample_no
        else:
            return bag_size, len(data) / bag_size
    else:
        if sample_no:
            return len(data) / sample_no, sample_no
        else:
            return len(data), 1


def partition(data, part_size, part_no):
    index = range(len(data))
    return [index[i:i + part_size] for i in xrange(0, len(index), part_size)][:part_no]


def bag(data, size, sample_no):
    n = len(data)
    data_n = []
    for i in range(sample_no):
        index = np.random.choice(n, size, True)
        data_n.append(index)
    return data_n


def cartesianVFKMM(tr_n, te_n):
    bags = []
    for tr in tr_n:
        for te in te_n:
            # string = str(tr) + "," + str(train[te].tolist()) + "," + str(test[te].tolist())
            # bags.append((tr, train[tr].tolist(), test[te].tolist()))
            bags.append((tr, tr, te))
    return bags


def cartesian(train, test, tr_n, te_n):
    bags = []
    for tr in tr_n:
        for te in te_n:
            bags.append((tr, train[tr], test[te]))
    return bags


def pair(train, test, tr_n, te_n, sample_no=10):
    bags = []
    tr_bag_indices = np.random.choice(len(tr_n), sample_no, replace=False)
    te_bag_indices = np.random.choice(len(te_n), sample_no, replace=False)
    for i in range(sample_no):
        tr_idx = tr_n[tr_bag_indices[i]]
        te_idx = te_n[te_bag_indices[i]]
        bags.append((tr_idx, train[tr_idx], test[te_idx]))
    return bags


import numpy as np
import math


def get_size_no(data, bag_size, sample_no):
    if bag_size:
        if sample_no:
            return bag_size, sample_no
        else:
            return bag_size, len(data) / bag_size
    else:
        if sample_no:
            return len(data) / sample_no, sample_no
        else:
            return len(data), 1


def partition(data, part_size, part_no):
    index = range(len(data))
    return [index[i:i + part_size] for i in xrange(0, len(index), part_size)][:part_no]


def bag(data, size, sample_no):
    n = len(data)
    data_n = []
    for i in range(sample_no):
        index = np.random.choice(n, size, True)
        data_n.append(index)
    return data_n


def cartesianKMM(train, test, tr_n, te_n):
    bags = []
    for tr in tr_n:
        for te in te_n:
            # string = str(tr) + "," + str(train[te].tolist()) + "," + str(test[te].tolist())
            # bags.append((tr, train[tr].tolist(), test[te].tolist()))
            bags.append((tr, tr, test[te]))
            # string = str(tr) + "," + str(te)
            # file = open(output_file, 'a')
            # file.write(string)
            # file.write('\n')
    return bags


def cartesian(train, test, tr_n, te_n):
    bags = []
    for tr in tr_n:
        for te in te_n:
            bags.append((tr, train[tr], test[te]))
    return bags


def pair(train, test, tr_n, te_n, sample_no=10):
    bags = []
    tr_bag_indices = np.random.choice(len(tr_n), sample_no, replace=False)
    te_bag_indices = np.random.choice(len(te_n), sample_no, replace=False)
    for i in range(sample_no):
        tr_idx = tr_n[tr_bag_indices[i]]
        te_idx = te_n[te_bag_indices[i]]
        bags.append((tr_idx, train[tr_idx], test[te_idx]))
    return bags

