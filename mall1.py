#!/usr/bin/env python
# encoding: utf-8
import glob
import scipy.io as sio
import json
import numpy as np
import cv2
from lib import get_dirs, read_img, get_feat, perform_regression

def get_cnt():
    npzfile = np.load(cfg['mall']['cvt_feat'])
    return npzfile['cnt']

def imgs_resize(imgs, sz):
    ret = []
    for img in imgs:
        ret.append(cv2.resize(img, sz))
    return np.asarray(ret)
    
def get_data():
    data_dir = [cfg['mall']['data_path']]
    imgs = read_img(data_dir, 'jpg')
    segms = np.empty_like(imgs, dtype=np.uint8)
    dmap_mat = sio.loadmat(cfg['mall']['dmap_path'])
    dmap = dmap_mat['pMapN']
    roi = dmap_mat['roi'][0, 0]['mask']
    segms[:] = roi
    return get_feat(imgs, segms, dmap, 10), get_cnt()

def main():
    feat, cnt = get_data()
    print feat.shape, cnt.shape
    # plt.plot(feat[:, 4], cnt, '.'); plt.show()
    np.savez("mall1", feat=feat, cnt=cnt)
    feat = feat[:, [15]]
    perform_regression(feat, cnt)

if __name__ == '__main__':
    with open("config.json") as cfgfile:
        cfg = json.load(cfgfile)
    main()
