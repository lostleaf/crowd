#!/usr/bin/env python
# encoding: utf-8

import os
import json
import numpy as np
import scipy.io as sio
import glob
import cv2
import matplotlib.pyplot as plt
from itertools import izip, chain
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split, cross_val_predict


def get_dirs(path):
    dirs = [os.path.join(path, d) for d in os.listdir(path)]
    return [d for d in dirs if os.path.isdir(d)]


def read_img(dirs):
    ipaths = chain.from_iterable(glob.glob(d + "/*.png") for d in dirs)
    imgs = [cv2.imread(p, 0) for p in ipaths]
    return np.asarray(imgs)

def get_feat():
    data_dirs = get_dirs(cfg['vidf']['data_path'])[:20]
    segm_dirs = get_dirs(cfg['vidf']['segm_path'])[:20]
    imgs = read_img(data_dirs)
    segms = read_img(segm_dirs)
    extractor = FeatExtractor()
    feat = []
    # segmer = Segmer(imgs)
    for i, s in izip(imgs, segms):
        _, segm = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY)
        feat.append(extractor.extract(i, segm))
    feat = np.asarray(feat)
    return feat

def get_cnt():
    gt_paths = glob.glob(cfg['vidf']['gt_path'] + "/vidf1_33_*_frame_full.mat")
    mats = (sio.loadmat(path) for path in gt_paths)
    cnt = [f['id'][0, 0].shape[1] for mat in mats for f in mat['fgt']['frame'][0, 0][0]]
    return np.array(cnt)

def get_data():
    return get_feat(), get_cnt()

def regression(feat, cnt):
    regr = LinearRegression()
    cnt_pred = cross_val_predict(regr, feat, cnt, cv=5)
    print np.mean(np.abs(cnt_pred - cnt))

def main():
    feat, cnt = get_data()
    print feat.shape, cnt.shape
    plt.plot(feat[:, 4], cnt, '.'); plt.show()
    np.save("fast_cornor", feat[:, 3:])
    regression(feat, cnt)

if __name__ == '__main__':
    with open("config.json") as cfgfile:
        cfg = json.load(cfgfile)
    dmap_mat = sio.loadmat(cfg['vidf']['dmap_path'])
    dmap = dmap_mat['dmap']['pmapxy'][0, 0]
    dmap_sqrt = np.sqrt(dmap)
    main()
