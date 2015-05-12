#!/usr/bin/env python
# encoding: utf-8

import cv2
from sklearn import svm
from sklearn.preprocessing import StandardScaler, scale
from sklearn.pipeline import Pipeline
from lib import read_img, get_dirs
from expr_utils import load_dataset
import json
from sklearn.metrics import mean_absolute_error as mae
import math
import numpy as np
import matplotlib.pyplot as plt
from itertools import izip

def get_data():
    data_dirs = get_dirs(cfg['vidf']['data_path'])[:20]
    segm_dirs = get_dirs(cfg['vidf']['segm_path'])[:20]
    imgs = read_img(data_dirs)
    segms = read_img(segm_dirs)
    feat, cnt = load_dataset(cfg['vidf']['cvt_feat'])
    return imgs, segms, feat, cnt

def gen_plot(cnt, cnt_pred):
    for i in xrange(cnt.shape[0]):
        st = max(i - 100, 0)
        fig = plt.figure()
        plt.xlim(0, 100)
        plt.ylim(0, 50)
        plt.plot(cnt[st : i+1], label="ground truth")
        plt.plot(cnt_pred[st : i+1], label="estimated")
        plt.xticks([])
        plt.ylabel("crowd counting")
        plt.legend(loc='best')
        plt.savefig('plot/%d.png' % i, bbox_inches='tight')
        plt.close(fig)
        
def main():
    imgs, segms, feat, cnt = get_data()
    feat_segm = feat[:, :9]
    feat_edge = feat[:, 9:16]
    feat_glcm = feat[:, 17:29]
    feat_pts = np.load('fast_ucsd.npy')
    scaler = StandardScaler()
    regr = svm.SVR(C=1e6, gamma=1e-6)
    pregr = Pipeline([('scaler', scaler), ('svr', regr)])
    pregr.fit(feat, cnt)
    cnt_pred = pregr.predict(feat)
    print "MAE:", mae(cnt, cnt_pred)
    fourcc = cv2.cv.FOURCC(*'mp4v')
    vout = cv2.VideoWriter('demo.avi', fourcc, 20.0, (976, 376), True)
    for i, (segm, img, c, cp) in enumerate(izip(segms, imgs, cnt, np.round(cnt_pred))):
        img1 = np.full((img.shape[0] + 30, img.shape[1], 3), 255, dtype=np.uint8)
        img1[:img.shape[0], :, 0] = img
        img1[:img.shape[0], :, 1] = np.minimum(img + (segm > 0) * 30, 255).astype(np.uint8)
        img1[:img.shape[0], :, 2] = img
        img1[segm > 0][1] = 255
        img1 = cv2.resize(img1, (0, 0), fx=2, fy=2)
        cv2.putText(img1, "estimated: %d error: %d" % (cp, abs(c - cp)), (10, img.shape[0] * 2 + 25), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0))
        pimg = cv2.imread("plot/%d.png" % i, cv2.CV_LOAD_IMAGE_COLOR)
        pimg = cv2.resize(pimg, (0, 0), fx=0.65, fy=0.65)
        img2 = np.full((img1.shape[0], img1.shape[1] + 500, 3), 255, dtype=np.uint8)
        img2[:,:img1.shape[1]] = img1
        img2[15 : 15 + pimg.shape[0], img1.shape[1] + 25 : img1.shape[1] + 25 + pimg.shape[1]] = pimg
        vout.write(img2)
        # print img2.shape
        # cv2.imshow('frame', img2)
        # if cv2.waitKey(50) & 0xFF == ord('q'):
        #     break
    vout.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    with open("config.json") as cfgfile:
        cfg = json.load(cfgfile)
    main()
