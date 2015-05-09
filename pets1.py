#!/usr/bin/env python
# encoding: utf-8
import glob
import scipy.io as sio
import json
import numpy as np
import cv2
from lib import get_dirs, read_img, get_feat, perform_regression

def get_cnt():
    npzfile = np.load(cfg['pets']['cvt_feat'])
    return npzfile['cnt']

def imgs_resize(imgs, sz):
    ret = []
    for img in imgs:
        ret.append(cv2.resize(img, sz))
    return np.asarray(ret)
    

def get_data():
    data_dirs = get_dirs(cfg['pets']['data_path'])
    segm_dirs = get_dirs(cfg['pets']['segm_path'])
    imgs = imgs_resize(read_img(data_dirs, 'jpg'), (256, 192))
    segms = read_img(segm_dirs)
    dmap_mat = sio.loadmat(cfg['pets']['dmap_path'])
    dmap = dmap_mat['dmap']['pmapxy'][0, 0]
    return get_feat(imgs, segms, dmap), get_cnt()

def main():
    feat, cnt = get_data()
    print feat.shape, cnt.shape
    # plt.plot(feat[:, 4], cnt, '.'); plt.show()
    np.savez("pets1", feat=feat, cnt=cnt)
    perform_regression(feat, cnt)

if __name__ == '__main__':
    with open("config.json") as cfgfile:
        cfg = json.load(cfgfile)
    main()
