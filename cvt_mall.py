import numpy as np
import scipy.io as sio
import json

def main():
    with open('config.json') as cfgfile: 
        cfg=json.load(cfgfile)
    feat = sio.loadmat(cfg['mall']["ori_feat"])['x']
    cnt = sio.loadmat(cfg['mall']['ori_cnt'])['count'][:, 0]
    np.savez(cfg['mall']['cvt_feat'], feat=feat, cnt=cnt)

if __name__ == '__main__':
    main()
