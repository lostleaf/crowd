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
from sklearn.cross_validation import train_test_split

def get_dirs(path):
    dirs = [os.path.join(path, d) for d in os.listdir(path)]
    return [d for d in dirs if os.path.isdir(d)]

def extract(img, segm):
    area = np.sum(dmap[segm > 0])
    perimeter = np.sum(dmap_sqrt[cv2.Canny(segm, 0, 255) > 0])
    return np.array([area, perimeter])

def get_feat():
    data_dirs = get_dirs(cfg['vidf']['data_path'])[:20]
    segm_dirs = get_dirs(cfg['vidf']['segm_path'])[:20]
    img_paths = list(chain.from_iterable(glob.glob(d + "/*.png") for d in data_dirs))
    segm_paths = list(chain.from_iterable(glob.glob(d + "/*.png") for d in segm_dirs))
    feat = []
    for ip, sp in izip(img_paths, segm_paths):
        img = cv2.imread(ip, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        _, segm= cv2.threshold(cv2.imread(sp, cv2.CV_LOAD_IMAGE_GRAYSCALE), 0, 255, cv2.THRESH_BINARY)
        feat.append(extract(img, segm))
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
    feat_train, feat_test, cnt_train, cnt_test = train_test_split(feat, cnt, test_size=0.2)
    regr = LinearRegression()
    regr.fit(feat_train, cnt_train)
    cnt_pred = regr.predict(feat_test)
    print np.mean(np.abs(cnt_pred - cnt_test))

def main():
    feat, cnt = get_data()
    print feat.shape, cnt.shape
    regression(feat, cnt)

if __name__ == '__main__':
    with open("config.json") as cfgfile:
        cfg = json.load(cfgfile)
    dmap_mat = sio.loadmat(cfg['vidf']['dmap_path'])
    dmap = dmap_mat['dmap']['pmapxy'][0, 0]
    dmap_sqrt = np.sqrt(dmap)
    main()
