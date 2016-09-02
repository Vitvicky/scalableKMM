from __future__ import division
import numpy as np
import math

def get_train_info(train_data, sample_no, eta):
    train_size = len(train_data)
    tr_bsize = int(train_size/sample_no)
    temp = 1 - 1 / train_size
    m = (math.log(eta,math.e))/(tr_bsize*math.log(temp,math.e))

    return int(tr_bsize),int(m)