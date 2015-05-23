#!/usr/bin/env python
# encoding: utf-8

import cv2
import numpy as np
import os
import glob
import scipy.io as sio
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split, cross_val_predict
from skimage.feature import greycomatrix, greycoprops, canny
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

    def __init__(self, dmap, fast_param):
        self.fast = cv2.FastFeatureDetector(fast_param)
        self.dmap = dmap
        self.dmap_sqrt = np.sqrt(dmap)
        self.kers = get_gaussians()
        # self.surf = cv2.SURF(400)

    def get_points(self, det, img, segm):
        points = det.detect(img, (segm > 0).astype(np.uint8))
        ret = 0
        for p in points:
            px, py = int(p.pt[0]), int(p.pt[1])
            ret += self.dmap_sqrt[py, px]
        return np.array([ret])      

    def hist_ori(self, img):
        # print img
        fimgs = []
        for i, k in enumerate(self.kers):
            fimgs.append(cv2.filter2D(img, -1, k))
        ori = np.argmax(np.array(fimgs), axis=0)
        x, y = np.nonzero(img)
        hist = []
        for i in xrange(6):
            idx = np.nonzero(ori[x, y] == i)
            nx, ny = x[idx], y[idx]
            hist.append(np.sum(self.dmap_sqrt[nx, ny]))
        return np.asarray(hist)
    
    def extract_segm(self, img, segm):
        area = np.sum(self.dmap[segm > 0])
        img_perimeter = cv2.Canny(segm, 0, 255)
        cnt_perimeter = np.sum(self.dmap_sqrt[img_perimeter > 0])
        histori_peri = self.hist_ori(img_perimeter)

        # print np.count_nonzero(img_perimeter > 0)
        return np.concatenate((np.array([area, cnt_perimeter]), histori_peri))
    
    def extract_edge(self, img, segm):
        img_edge = cv2.Canny(img, 100, 200)
        # img_edge = canny(img, sigma=3, low_threshold=100, high_threshold=200).astype(np.uint8)
        img_edge[segm == 0] = 0
        cnt_edge = np.sum(self.dmap_sqrt[img_edge > 0])
        histori_edge = self.hist_ori(img_edge)
        return np.concatenate((np.array([cnt_edge]), histori_edge))

    def extract_glcm(self, img, segm):
        img1 = img / 32
        img1[segm > 0] += 1
        img1[segm == 0] = 0
        g = greycomatrix(img1, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=9)[1:, 1:, :, :]
        gg = g / float(np.sum(g))
        homo = greycoprops(gg, prop="homogeneity")[0]
        energy = greycoprops(gg, prop="energy")[0]
        gg[gg == 0] = 1
        entropy = -np.sum(np.multiply(gg, np.log(gg)), axis=(0,1))[0]
        return np.concatenate((homo, energy, entropy))
        
    def extract(self, img, segm):
        feat_segm = self.extract_segm(img, segm)
        feat_edge = self.extract_edge(img, segm)
        feat_corner = self.get_points(self.fast, img, segm)
        feat_glcm = self.extract_glcm(img, segm)
        return np.concatenate((feat_segm, feat_edge, feat_corner, feat_glcm))

def perform_regression(feat, cnt):
    regr = LinearRegression()
    cnt_pred = cross_val_predict(regr, feat, cnt, cv=5)
    print np.mean(np.abs(cnt_pred - cnt))
    print np.mean(np.abs(cnt_pred - cnt) / cnt)

def get_feat(imgs, segms, dmap, fast_param=40):
    extractor = FeatExtractor(dmap, fast_param)
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

def get_gaussians(sigmax=4, sigmay=.1):
    kx = cv2.getGaussianKernel(17, sigmax)
    ky = cv2.getGaussianKernel(17, sigmay)
    ker = np.dot(ky, kx.T)
    rows,cols = ker.shape
    kers = [ker]
    for deg in np.linspace(30, 150, 5):
        m = cv2.getRotationMatrix2D((cols/2,rows/2),deg,1)
        kers.append(cv2.warpAffine(ker, m, (cols, rows)))
    return kers

def test():
    img = cv2.imread("/Volumes/Untitled/crowd counting/ucsd/ucsdpeds_vidf/video/vidf/vidf1_33_000.y/vidf1_33_000_f001.png", 0)
    segm = cv2.imread("/Volumes/Untitled/crowd counting/ucsd/ucsdpeds_vidf/segm/vidf/vidf1_33_000.segm/vidf1_33_000_f001.png", 0)
    dmap = sio.loadmat("/Volumes/Untitled/crowd counting/ucsd/ucsdpeds_gt/gt/vidf/vidf1_33_dmap3.mat")['dmap']['pmapxy'][0, 0]
    extractor = FeatExtractor(dmap) 
    print extractor.extract(img, segm)

if __name__ == '__main__':
    test() 
