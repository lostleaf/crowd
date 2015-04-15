import numpy as np
import scipy.io as sio
import json


def main():
    with open('config.json') as cfgfile:
        cfg = json.load(cfgfile)
    mat = sio.loadmat(cfg['pets']["ori_feat"])
    # print mat.keys()
    feat1 = np.concatenate(mat['fv'][0, 0][0, :], axis=1).T
    feat2 = np.concatenate(mat['fv'][0, 1][0, :], axis=1).T
    cnt1 = np.concatenate(mat['cnt'][0, 0][0, :], axis=1).T[:, 0]
    cnt2 = np.concatenate(mat['cnt'][0, 1][0, :], axis=1).T[:, 0]
    cnt = cnt1 + cnt2
    feat = feat1 + feat2
    print feat.shape
    # print np.count_nonzero(cnt1 + cnt2)
    np.savez(cfg['pets']['cvt_feat'], feat=feat, cnt=cnt)
    # print cnt1.shape

if __name__ == '__main__':
    main()
