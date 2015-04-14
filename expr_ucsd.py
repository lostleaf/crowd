import numpy as np
import json
from sklearn import linear_model
from sklearn import kernel_ridge
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt
from expr_utils import filter_by_ranges, expr, load_dataset

def linear(feat_train, cnt_train, feat_test, cnt_test):
    regr = linear_model.LinearRegression()
    expr(regr,  "linear", feat_train, cnt_train, feat_test, cnt_test)

def ridge(feat_train, cnt_train, feat_test, cnt_test):
    regr = linear_model.Ridge(alpha=1e-3)
    expr(regr, "ridge", feat_train, cnt_train, feat_test, cnt_test)

def lkridge(feat_train, cnt_train, feat_test, cnt_test):
    regr = kernel_ridge.KernelRidge(alpha=1e-3)
    expr(regr,  "linear kernel ridge regression", feat_train, cnt_train, feat_test, cnt_test)

def lksvr(feat_train, cnt_train, feat_test, cnt_test):
    scaler = StandardScaler()
    regr = svm.LinearSVR(C=1e-1)
    pipeline = Pipeline([('scaler', scaler), ('svr', regr)])
    expr(pipeline,  "linear kernel SVR", feat_train, cnt_train, feat_test, cnt_test)

def rbfsvr(feat_train, cnt_train, feat_test, cnt_test):
    scaler = StandardScaler()
    regr = svm.SVR(C=1e2, gamma=0.0002)
    pipeline = Pipeline([('scaler', scaler), ('svr', regr)])
    expr(pipeline,  "rbf kernel SVR", feat_train, cnt_train, feat_test, cnt_test)

def main():
    with open('config.json') as cfg_file:
        cfg = json.load(cfg_file)['vidf']

    feat_train, cnt_train, feat_test, cnt_test = load_dataset(cfg['cvt_feat'], cfg['trainset'], cfg['testset'])

    # print feat_train.shape, feat_test.shape
    linear(feat_train, cnt_train, feat_test, cnt_test)
    ridge(feat_train, cnt_train, feat_test, cnt_test)
    lkridge(feat_train, cnt_train, feat_test, cnt_test)
    lksvr(feat_train, cnt_train, feat_test, cnt_test)
    rbfsvr(feat_train, cnt_train, feat_test, cnt_test)


if __name__ == '__main__':
    main()
