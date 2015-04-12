import cv2
import numpy as np
import config
import glob
from matplotlib import pyplot as plt

def main():
    iter_imgs = glob.iglob(config.IMG_PATH + "/*.png")
    # print(list(iter_imgs))
    imgs = np.array([cv2.imread(img_file, cv2.CV_LOAD_IMAGE_GRAYSCALE) for img_file in iter_imgs])
    # plt.imshow(imgs[0], cmap='gray')
    # plt.show()
    bg = np.median(imgs, axis=0)
    fg_mask = np.abs(imgs - bg) > 7
    print fg_mask.shape

    plt.subplot(211), plt.xticks([]), plt.yticks([])
    plt.imshow(imgs[100], cmap='gray')
    plt.subplot(212), plt.xticks([]), plt.yticks([])
    plt.imshow(fg_mask[100], cmap='Greys')
    plt.show()


if __name__ == '__main__':
    main()
