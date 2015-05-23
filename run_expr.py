#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import sys
import json
import importlib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_predict


def load_dataset(path):
    feat_cnt = np.load(path)
    feat, cnt = feat_cnt['feat'], feat_cnt['cnt']
    return feat, cnt

def my_import(name):
    module_name, class_name = name.rsplit('.', 1)
    mod = importlib.import_module(module_name)
    return getattr(mod, class_name)

def run(config, feat_groups, cnt):
    print "run experiment %s" % config['name']
    feat = np.concatenate([feat_groups[f] for f in config["use_feat"]], axis=1)
    print feat.shape
    m = my_import(config['model'])
    regr = m(**config['params'])
    if config['normalization']:
        regr = Pipeline([('scaler', StandardScaler()), ('regressor', regr)])

    cnt_pred = cross_val_predict(regr, feat, cnt, cv=5)
    print "MAE: %.2f" % np.mean(np.abs(cnt_pred  - cnt))
    print "MSE: %.2f" % np.mean(np.square(cnt_pred - cnt))
    print "MRE: %.2f%%" % (np.mean(np.abs(cnt_pred  - cnt) / cnt) * 100)

def main(argv):
    if len(argv) != 2:
        print "usage: python run_expr.py config_of_expr.json"
        sys.exit(-1)
    with open(argv[1], 'r') as cfg_file:
        cfg = json.load(cfg_file)
    feat_ori, cnt = load_dataset(cfg['dataset_path'])
    # print feat_ori.shape
    feat_groups = {}
    for k, v in cfg['feat_groups'].iteritems():
        if type(v[0]) == list:
            feat_groups[k] = np.concatenate([feat_ori[:, vv[0] : vv[1]]for vv in v], axis = 1)
        elif type(v[0]) == int:
            feat_groups[k] = feat_ori[:, v[0] : v[1]] 
    print feat_groups['segm'].shape
    for config in cfg["models"]:
        if config['work']:
            run(config, feat_groups, cnt)

if __name__ == '__main__':
    main(sys.argv)
