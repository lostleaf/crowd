import numpy as np
import json
from sklearn import linear_model
from sklearn import kernel_ridge
from sklearn import preprocessing
from sklearn import svm
from matplotlib import pyplot as plt
from expr_utils import filter_by_ranges, expr

def linear(feat_train, cnt_train, feat_test, cnt_test):
    regr = linear_model.LinearRegression()
    expr(regr,  "linear", feat_train, cnt_train, feat_test, cnt_test)

def ridge(feat_train, cnt_train, feat_test, cnt_test):
    regr = linear_model.Ridge(alpha=0.001)
    expr(regr, "ridge", feat_train, cnt_train, feat_test, cnt_test)

def lkridge(feat_train, cnt_train, feat_test, cnt_test):
    regr = kernel_ridge.KernelRidge(alpha=0.0008)
    expr(regr,  "linear kernel ridge regression", feat_train, cnt_train, feat_test, cnt_test)

def lksvr(feat_train, cnt_train, feat_test, cnt_test):
    regr = svm.LinearSVR(C=1000)
    expr(regr,  "linear kernel SVR", feat_train, cnt_train, feat_test, cnt_test)

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
    lkridge(feat_train, cnt_train, feat_test, cnt_test)
    lksvr(feat_train, cnt_train, feat_test, cnt_test)


if __name__ == '__main__':
    main()
