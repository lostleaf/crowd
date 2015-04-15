import numpy as np
import json
from sklearn import linear_model
from sklearn import kernel_ridge
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt
from expr_utils import filter_by_ranges, expr, load_dataset

with open('config.json') as cfg_file:
    cfg = json.load(cfg_file)['vidf']

feat, cnt = load_dataset(cfg['cvt_feat'])

def linear():
    regr = linear_model.LinearRegression()
    expr(regr,  "linear", feat, cnt)

def ridge():
    regr = linear_model.Ridge(alpha=1e-3)
    expr(regr, "ridge", feat, cnt)

def lkridge():
    regr = kernel_ridge.KernelRidge(alpha=1e2)
    expr(regr,  "linear kernel ridge regression", feat, cnt)

def lksvr():
    scaler = StandardScaler()
    regr = svm.LinearSVR(C=0.7)
    pipeline = Pipeline([('scaler', scaler), ('svr', regr)])
    expr(pipeline,  "linear kernel SVR", feat, cnt)

def rbfsvr():
    scaler = StandardScaler()
    regr = svm.SVR(C=7e2, gamma=0.0002)
    pipeline = Pipeline([('scaler', scaler), ('svr', regr)])
    expr(pipeline,  "rbf kernel SVR", feat, cnt)

def main():
    # print feat_train.shape, feat_test.shape
    linear()
    ridge()
    lkridge()
    lksvr()
    rbfsvr()


if __name__ == '__main__':
    main()
