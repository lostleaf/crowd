#!/usr/bin/env python
# encoding: utf-8

import os
import json
import numpy as np
import scipy.io as sio
import glob
import cv2
import matplotlib.pyplot as plt
from itertools import izip
from sklearn.linear_model import LinearRegression

def get_dirs(path):
    dirs = [os.path.join(path, d) for d in os.listdir(path)]
    return [d for d in dirs if os.path.isdir(d)]

def extract(img, segm_mask):
    return np.array([np.count_nonzero(segm_mask)])


def main():
    with open("config.json") as cfgfile:
        cfg = json.load(cfgfile)
    data_dirs = get_dirs(cfg['vidf']['data_path'])
    segm_dirs = get_dirs(cfg['vidf']['segm_path'])
    # print data_dirs[0], segm_dirs[0]
    img_paths = glob.glob(data_dirs[0] + "/*.png")
    segm_paths = glob.glob(segm_dirs[0] + "/*.png")
    feats = []
    for ip, sp in izip(img_paths, segm_paths):
        img = cv2.imread(ip, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        segm_img = cv2.imread(sp, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        segm_mask = segm_img > 0
        feats.append(extract(img, segm_mask))
    feats = np.asarray(feats)


if __name__ == '__main__':
    main()
