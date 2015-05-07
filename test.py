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

class Segmentation(object):

    def __init__(self, imgs):
        self.bg = np.median(imgs, axis=0)
        # plt.imshow(self.bg, cmap='gray')
        # plt.show()

    def segm(self, imgs):
        fg_mask = np.abs(imgs - self.bg) > 10
        return fg_mask
class FeatExtractor(object):

    def __init__(self):
        self.fast = cv2.FastFeatureDetector(40)
        # self.surf = cv2.SURF(400)

    def get_points(self, det, img, segm):
        points = det.detect(img, (segm > 0).astype(np.uint8))
        ret = 0
        for p in points:
            px, py = int(p.pt[0]), int(p.pt[1])
            ret += dmap_sqrt[py, px]
        return ret      

    def extract(self, img, segm):
        # segm1 = segmer.segm(img)
        area = np.sum(dmap[segm > 0])
        perimeter = np.sum(dmap_sqrt[cv2.Canny(segm, 0, 255) > 0])
        edge = np.sum(dmap_sqrt[np.logical_and(cv2.Canny(img, 100, 200) > 0, segm > 0)])
        # px, py = self.get_fast_points(img, segm)
        pts_fast = self.get_points(self.fast, img, segm)
        # pts_surf = self.get_points(self.surf, img, segm)
        return np.array([area, perimeter, edge, pts_fast])

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
