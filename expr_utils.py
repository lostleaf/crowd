#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from sklearn.preprocessing import scale
from sklearn.cross_validation import cross_val_predict

def load_dataset(path):
    feat_cnt = np.load(path)
    feat, cnt = feat_cnt['feat'], feat_cnt['cnt']
    return feat, cnt

def filter_by_ranges(arr, rngs):
    data_list = [arr[st:en] for st, en in rngs]
    return np.concatenate(data_list, axis=0)

def expr(regr, name, feat, cnt):
    print name
    cnt_pred = cross_val_predict(regr, feat, cnt, cv=5)
    # print cnt_pred.shape
    # regr.fit(feat_train, cnt_train)
    # cnt_pred = regr.predict(feat_test)
    print "MAE: %.2f" % np.mean(np.abs(cnt_pred  - cnt))
    print "MSR: %.2f" % np.mean(np.square(cnt_pred - cnt))
    print "MRE: %.2f%%" % (np.mean(np.abs(cnt_pred  - cnt) / cnt) * 100)

