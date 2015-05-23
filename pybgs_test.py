import numpy as np
import cv2
import pybgs
import glob
import lib

params = { 
	'algorithm': 'zivkovic_agmm', 
	'low': 5,
	'high': 20,
	'alpha': 0.02,
	'max_modes': 3 }

bg_sub = pybgs.BackgroundSubtraction()	
path = "/Volumes/Untitled/crowd counting/mall/frames"
path = "/Volumes/Untitled/crowd counting/ucsd/ucsdpeds_vidf/video/vidf/vidf1_33_000.y"
img_paths = glob.glob(path + '/*.png')
img = cv2.imread(img_paths[0], cv2.CV_LOAD_IMAGE_COLOR)

high_threshold_mask = np.zeros(shape=img.shape[0:2], dtype=np.uint8)
low_threshold_mask = np.zeros_like(high_threshold_mask)
bg_sub.init_model(img, params)

for i, img_path in enumerate(img_paths[1:]):
    img = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_COLOR) 
    bg_sub.subtract(i, img, low_threshold_mask, high_threshold_mask)
    bg_sub.update(i, img, high_threshold_mask)
    cv2.imshow('foreground', low_threshold_mask)
    # cv2.imshow('background', bg_sub.get_background())
    cv2.waitKey(40)
