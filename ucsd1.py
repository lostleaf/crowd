#!/usr/bin/env python
# encoding: utf-8

import os
import json
import numpy as np
import scipy.io as sio
import cv2
import glob
import matplotlib.pyplot as plt
from itertools import izip
from lib import FeatExtractor, perform_regression, get_feat, get_dirs, read_img

def get_cnt():
    gt_paths = glob.glob(cfg['vidf']['gt_path'] + "/vidf1_33_*_frame_full.mat")
    mats = (sio.loadmat(path) for path in gt_paths)
    cnt = [f['id'][0, 0].shape[1] for mat in mats for f in mat['fgt']['frame'][0, 0][0]]
    return np.array(cnt)

def get_data():
    data_dirs = get_dirs(cfg['vidf']['data_path'])[:20]
    segm_dirs = get_dirs(cfg['vidf']['segm_path'])[:20]
    imgs = read_img(data_dirs)
    segms = read_img(segm_dirs)
    dmap_mat = sio.loadmat(cfg['vidf']['dmap_path'])
    dmap = dmap_mat['dmap']['pmapxy'][0, 0]
    return get_feat(imgs, segms, dmap), get_cnt()

def main():
    feat, cnt = get_data()
    print feat.shape, cnt.shape
    # plt.plot(feat[:, 4], cnt, '.'); plt.show()
    np.save("fast_ucsd", feat[:, [15]])
    perform_regression(feat, cnt)

if __name__ == '__main__':
    with open("config.json") as cfgfile:
        cfg = json.load(cfgfile)
    main()
