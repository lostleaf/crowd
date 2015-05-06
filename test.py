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

class Segmer(object):

    def __init__(self, imgs):
        self.bg = np.median(imgs, axis=0)
        # plt.imshow(self.bg, cmap='gray')
        # plt.show()

    def segm(self, imgs):
        fg_mask = np.abs(imgs - self.bg) > 10
        return fg_mask

def get_dirs(path):
    dirs = [os.path.join(path, d) for d in os.listdir(path)]
    return [d for d in dirs if os.path.isdir(d)]

def extract(img, segm):
    # segm1 = segmer.segm(img)
    area = np.sum(dmap[segm > 0])
    perimeter = np.sum(dmap_sqrt[cv2.Canny(segm, 0, 255) > 0])
    edge = np.sum(dmap_sqrt[np.logical_and(cv2.Canny(img, 100, 200) > 0, segm > 0)])
    return np.array([area, perimeter, edge])

def read_img(dirs):
    ipaths = chain.from_iterable(glob.glob(d + "/*.png") for d in dirs)
    imgs = [cv2.imread(p, 0) for p in ipaths]
    return np.asarray(imgs)

def get_feat():
    data_dirs = get_dirs(cfg['vidf']['data_path'])[:20]
    segm_dirs = get_dirs(cfg['vidf']['segm_path'])[:20]
    imgs = read_img(data_dirs)
    segms = read_img(segm_dirs)
    feat = []
    # segmer = Segmer(imgs)
    for i, s in izip(imgs, segms):
        _, segm = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY)
        feat.append(extract(i, segm))
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
    plt.plot(feat[:, 2], cnt, '.'); plt.show()
    regression(feat, cnt)

if __name__ == '__main__':
    with open("config.json") as cfgfile:
        cfg = json.load(cfgfile)
    dmap_mat = sio.loadmat(cfg['vidf']['dmap_path'])
    dmap = dmap_mat['dmap']['pmapxy'][0, 0]
    dmap_sqrt = np.sqrt(dmap)
    main()
