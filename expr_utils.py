#!/usr/bin/env python
# encoding: utf-8

import numpy as np

def filter_by_ranges(arr, rngs):
    data_list = [arr[st:en] for st, en in rngs]
    return np.concatenate(data_list, axis=0)

def expr(regr, name, feat_train, cnt_train, feat_test, cnt_test):
    regr.fit(feat_train, cnt_train)
    cnt_pred = regr.predict(feat_test)
    print name
    print "MAE: %.2f" % np.mean(np.abs(cnt_pred  - cnt_test))
    print "MRE: %.2f%%" % (np.mean(np.abs(cnt_pred  - cnt_test) / cnt_test) * 100)

