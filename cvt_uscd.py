import numpy as np
import scipy.io as sio
import json

def main():
    with open('config.json') as cfgfile: 
        cfg=json.load(cfgfile)
    mat = sio.loadmat(cfg['vidf']["ori_feat"])
    feat = mat['fv'][0,2].T
    cnt = mat['cnt'][0,2][0]
    np.savez(cfg['vidf']['cvt_feat'], feat=feat, cnt=cnt)

if __name__ == '__main__':
    main()
