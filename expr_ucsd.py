import numpy as np
import json
from sklearn import linear_model
from sklearn import kernel_ridge
from sklearn import preprocessing
from matplotlib import pyplot as plt

def filter_by_ranges(arr, rngs):
    data_list = [arr[st:en] for st, en in rngs]
    return np.concatenate(data_list, axis=0)

def linear(feat_train, cnt_train, feat_test, cnt_test):
    regr = linear_model.LinearRegression()
    regr.fit(feat_train, cnt_train)
    cnt_pred = regr.predict(feat_test)
    print "linear"
    print "MAE: %.2f" % np.mean(np.abs(cnt_pred  - cnt_test))
    print "MRE: %.2f%%" % (np.mean(np.abs(cnt_pred  - cnt_test) / cnt_test) * 100)

def ridge(feat_train, cnt_train, feat_test, cnt_test):
    regr = linear_model.Ridge(alpha=0.0008)
    regr.fit(feat_train, cnt_train)
    cnt_pred = regr.predict(feat_test)
    print "ridge"
    print "MAE: %.2f" % np.mean(np.abs(cnt_pred  - cnt_test))
    print "MRE: %.2f%%" % (np.mean(np.abs(cnt_pred  - cnt_test) / cnt_test) * 100)

def kridge(feat_train, cnt_train, feat_test, cnt_test):
    regr = kernel_ridge.KernelRidge(alpha=0.0008)
    regr.fit(feat_train, cnt_train)
    cnt_pred = regr.predict(feat_test)
    print "linear kernel ridge"
    print "MAE: %.2f" % np.mean(np.abs(cnt_pred  - cnt_test))
    print "MRE: %.2f%%" % (np.mean(np.abs(cnt_pred  - cnt_test) / cnt_test) * 100)

def main():
    with open('config.json') as cfg_file:
        cfg = json.load(cfg_file)['vidf']
    
    feat_cnt = np.load(cfg['cvt_feat'])
    feat, cnt = feat_cnt['feat'], feat_cnt['cnt']

    feat_train = filter_by_ranges(feat, cfg['trainset'])
    cnt_train = filter_by_ranges(cnt, cfg['trainset'])
    feat_test = filter_by_ranges(feat, cfg['testset'])
    cnt_test = filter_by_ranges(cnt, cfg['testset'])

    # print feat_train.shape, feat_test.shape
    linear(feat_train, cnt_train, feat_test, cnt_test)
    ridge(feat_train, cnt_train, feat_test, cnt_test)

    kridge(feat_train, cnt_train, feat_test, cnt_test)


if __name__ == '__main__':
    main()
