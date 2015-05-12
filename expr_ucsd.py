import numpy as np
import json
import math
from sklearn import linear_model
from sklearn import kernel_ridge
from sklearn import svm
from sklearn.preprocessing import StandardScaler, scale
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcess
from matplotlib import pyplot as plt
from expr_utils import filter_by_ranges, expr, load_dataset
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor

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
    regr = svm.SVR(C=1e6, gamma=1e-6)
    pipeline = Pipeline([('scaler', scaler), ('svr', regr)])
    expr(pipeline,  "rbf kernel SVR", feat, cnt)

def ker(x, y):
    s = x - y
    return np.dot(x, y) #+ math.exp(-np.dot(s, s))

def gpr():
    scaler = StandardScaler()
    regr = GaussianProcess(regr='linear', corr='linear', theta0=0.3)
    pipeline = Pipeline([('scaler', scaler), ('gpr', regr)])
    expr(pipeline,  "GPR", scale(feat), cnt)

def main():
    # print feat_train.shape, feat_test.shape
    linear()
    # ridge()
    # lkridge()
    # lksvr()
    rbfsvr()
    # gpr()


if __name__ == '__main__':
    with open('config.json') as cfg_file:
        cfg = json.load(cfg_file)['vidf']

    feat, cnt = load_dataset(cfg['cvt_feat'])
    # feat = feat[:, [0,1,9]]
    feat_segm = feat[:, :9]
    feat_edge = feat[:, 9:16]
    feat_glcm = feat[:, 17:29]
    feat_pts = np.load('fast_ucsd.npy')
    feat = np.concatenate((feat_segm, feat_edge, feat_pts), axis=1)
    
    main()
