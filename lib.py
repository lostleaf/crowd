#!/usr/bin/env python
# encoding: utf-8

import cv2
import numpy as np
import os
import glob
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split, cross_val_predict
from itertools import chain, izip

class Segmentation(object):

    def __init__(self, imgs):
        self.bg = np.median(imgs, axis=0)
        # plt.imshow(self.bg, cmap='gray')
        # plt.show()

    def segm(self, imgs):
        fg_mask = np.abs(imgs - self.bg) > 10
        return fg_mask
class FeatExtractor(object):

    def __init__(self, dmap):
        self.fast = cv2.FastFeatureDetector(40)
        self.dmap = dmap
        self.dmap_sqrt = np.sqrt(dmap)
        # self.surf = cv2.SURF(400)

    def get_points(self, det, img, segm):
        points = det.detect(img, (segm > 0).astype(np.uint8))
        ret = 0
        for p in points:
            px, py = int(p.pt[0]), int(p.pt[1])
            ret += self.dmap_sqrt[py, px]
        return ret      

    def extract(self, img, segm):
        # segm1 = segmer.segm(img)
        area = np.sum(self.dmap[segm > 0])
        perimeter = np.sum(self.dmap_sqrt[cv2.Canny(segm, 0, 255) > 0])
        edge = np.sum(self.dmap_sqrt[np.logical_and(cv2.Canny(img, 100, 200) > 0, segm > 0)])
        # px, py = self.get_fast_points(img, segm)
        pts_fast = self.get_points(self.fast, img, segm)
        # pts_surf = self.get_points(self.surf, img, segm)
        return np.array([area, perimeter, edge, pts_fast])

def perform_regression(feat, cnt):
    regr = LinearRegression()
    cnt_pred = cross_val_predict(regr, feat, cnt, cv=5)
    print np.mean(np.abs(cnt_pred - cnt))

def get_feat(imgs, segms, dmap):
    extractor = FeatExtractor(dmap)
    feat = []
    # segmer = Segmer(imgs)
    for i, s in izip(imgs, segms):
        _, segm = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY)
        feat.append(extractor.extract(i, segm))
    feat = np.asarray(feat)
    return feat

def get_dirs(path):
    dirs = [os.path.join(path, d) for d in os.listdir(path)]
    return [d for d in dirs if os.path.isdir(d)]

def read_img(dirs, ext='png'):
    ipaths = chain.from_iterable(glob.glob(d + '/*.' + ext) for d in dirs)
    imgs = [cv2.imread(p, 0) for p in ipaths]
    return np.asarray(imgs)

