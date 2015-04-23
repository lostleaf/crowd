import numpy as np
import json
from sklearn import linear_model
from sklearn import kernel_ridge
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt
from expr_utils import filter_by_ranges, expr, load_dataset
from sklearn.gaussian_process import GaussianProcess

with open('config.json') as cfg_file:
    cfg = json.load(cfg_file)['pets']

feat, cnt = load_dataset(cfg['cvt_feat'])


def linear():
    regr = linear_model.LinearRegression()
    expr(regr,  "linear", feat, cnt)


def ridge():
    regr = linear_model.Ridge(alpha=18)
    expr(regr, "ridge", feat, cnt)


def lkridge():
    regr = kernel_ridge.KernelRidge(alpha=1e1)
    expr(regr,  "linear kernel ridge regression", feat, cnt)


def lksvr():
    scaler = StandardScaler()
    regr = svm.LinearSVR(C=0.1)
    pipeline = Pipeline([('scaler', scaler), ('svr', regr)])
    expr(pipeline,  "linear kernel SVR", feat, cnt)


def rbfsvr():
    scaler = StandardScaler()
    regr = svm.SVR(C=1300, gamma=1e-5)
    pipeline = Pipeline([('scaler', scaler), ('svr', regr)])
    expr(pipeline,  "rbf kernel SVR", feat, cnt)


def gpr():
    scaler = StandardScaler()
    regr = GaussianProcess(regr='constant', corr='cubic', theta0=0.07)
    pipeline = Pipeline([('scaler', scaler), ('gpr', regr)])
    expr(pipeline,  "GPR", feat, cnt)



def main():
    linear()
    ridge()
    lksvr()
    rbfsvr()
    gpr()

if __name__ == '__main__':
    main()
