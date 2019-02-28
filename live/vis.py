N_CLF = 100

W = {
    "RJAA": [
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 5]
    ]
}


def predict(X, clfs, w, smooth=False, confidence=False):
    import numpy as np
    n_clf = len(clfs)
    p_rf = np.zeros((len(X), n_clf))

    for i, clf in enumerate(clfs):
        p_rf[:, i] = clf.predict(X.values)

    # p = p_rf.mean(axis=1)
    # p = majority_vote(p_rf, n_class=9, threshold=None)

    p = deflected_mean(p_rf, w, threshold=6)
    if smooth:
        p = smoothing(p, ksize=6, threshold=5)

    if confidence:
        c = confidence_factor(p_rf, n_class=9)
        return p, c

    else:
        return p


def confidence_factor(x, n_class):
    import numpy as np
    import skynet.nwp2d as npd
    mv = np.zeros((len(x), n_class))
    idx = np.arange(len(x))
    for i in range(x.shape[1]):
        mv[idx, x[:, i].astype(int)] += 1

    confac = npd.NWPFrame(mv)

    return confac


def deflected_mean(x, w, threshold=None):
    import numpy as np
    xx = np.zeros_like(x)
    mp = np.zeros_like(x)
    ww = np.array([w for _ in range(len(x))])
    idx = np.arange(len(x))
    for i in range(x.shape[1]):
        mp[:, i] = ww[idx, x[:, i].astype(int)]
        xx[:, i] = x[:, i] * mp[:, i]

    p = xx.sum(axis=1) / mp.sum(axis=1)

    if threshold is not None:
        p[p > threshold] = p.max()

    return p


def smoothing(x, ksize, threshold=None):
    import numpy as np
    kernel = np.ones(ksize)
    x_sm = 1 / kernel.sum() * np.convolve(x, kernel, mode="same")

    if threshold is not None:
        extend = np.zeros_like(x)
        for i in range(len(x)):
            idx1 = i - int(ksize / 2)
            idx2 = i + int(ksize / 2)
            if x_sm[i] < threshold:
                if idx1 >= 0 and idx2 <= len(x):
                    extend[i] = x_sm[i] * x[idx1:idx2].min() / (x_sm[idx1:idx2].min() + 1e-2)
                elif idx1 < 0:
                    extend[i] = x_sm[i] * x[:idx2].min() / (x_sm[:idx2].min() + 1e-2)
                elif idx2 > 0:
                    extend[i] = x_sm[i] * x[idx1:].min() / (x_sm[idx1:].min() + 1e-2)

            else:
                extend[i] = x_sm[i]

        extend[extend > 6] = 8
        return extend

    return x_sm


def get_month_key(date, period=2):
    keys = [
        (0, "month:1-2"),
        (1, "month:3-4"),
        (2, "month:5-6"),
        (3, "month:7-8"),
        (4, "month:9-10"),
        (5, "month:11-12")
    ]
    i = 4
    e = 6
    d = int(date[i:e])

    if d % period == 0:
        idx = int(d / period) - 1
    else:
        idx = int(d / period)

    return keys[idx]


def adapt_visibility(v):
    import copy
    import numpy as np
    import skynet.datasets as skyds
    v = copy.deepcopy(v)
    vis_level = skyds.base.get_init_vis_level()
    diff = np.diff(list(vis_level.values()) + [9999])
    for key in vis_level:
        v[(v > key) & (v <= (key + 1))] = \
            diff[key] * (v[(v > key) & (v <= (key + 1))] - key) + vis_level[key]
    v[v >= list(vis_level.values())[-1]] = 9999
    return v


def Vis_Pred(model, contxt, lclid, test_dir, input_dir, fit_dir, pred_dir, errfile):
    import os
    import sys
    import copy
    import csv
    import pickle
    import pandas as pd
    import skynet.nwp2d as npd
    import skynet.datasets as skyds
    from sklearn.preprocessing import StandardScaler
    from pathlib import Path

    myname = sys.argv[0]

    print(model)

    csv_test = '%s/%s-%s.csv' % (test_dir, contxt, lclid)
    csv_input = '%s/%s-%s.vis.csv' % (input_dir, contxt, lclid)
    fitfile = '%s/%s-%s.vis.pkl' % (fit_dir, contxt, lclid)
    predfile = '%s/%s-%s.vis.csv' % (pred_dir, contxt, lclid)
    conffile = '%s/confidence_factor/%s-%s.vis.csv' % (pred_dir, contxt, lclid)

    if not os.path.exists(csv_test):
        print("{:s}: [Error] {:s} is not found !".format(myname, csv_test))

        if not os.path.exists(errfile):
            Path(errfile).touch()

        return

    X = pd.read_csv(csv_test)
    X = npd.NWPFrame(X)

    # --- Reading Fitting File & Input File (If Not Existing -> -9999.)
    if not os.path.exists(fitfile) or not os.path.exists(csv_input):
        print("{:s}: [Checked] {:s} or {:s} is not found !".format(myname, fitfile, csv_input))
        PRED = []
        for k in range(len(X)):
            pred = [-9999.]
            PRED = PRED + pred

        # - Output(all -9999.)
        outdata = X[['HEAD:YEAR', 'MON', 'DAY', 'HOUR']]
        outdata['SKYNET-VIS'] = PRED
        outdata.to_csv(predfile, columns=['HEAD:YEAR', 'MON', 'DAY', 'HOUR', 'ARC-GUSTS'], index=False, header=True)

        # - Output(num of train -> 0)
        f = open(predfile, 'a')
        csv.writer(f, lineterminator='\n').writerow(['FOOT:TRAIN_NUM', 0])
        f.close()
        return

    df_date = X[['HEAD:YEAR', 'MON', 'DAY', 'HOUR']]
    date_keys = ['HEAD:YEAR', 'MON', 'DAY', 'HOUR', 'MIN']
    X['MIN'] = [0] * len(X)
    for key in date_keys:
        if not key == 'HEAD:YEAR':
            X[key] = ['%02d' % int(d) for d in X[key]]

    X.merge_strcol(date_keys, 'date', inplace=True)
    X.drop(date_keys, axis=1, inplace=True)

    # print(X)
    wni_code = skyds.get_init_features('wni')
    X = X[wni_code]

    long_code = skyds.get_init_features('long')
    X.columns = long_code

    vt = len(X)

    pool = skyds.read_csv(csv_input)[long_code]
    sppool = skyds.convert.split_time_series(pool, date=pool["date"].values, level="month", period=2,
                                             index_date=True)

    month_key_info = get_month_key(X['date'][0], period=2)
    X = pd.concat([X, sppool[month_key_info[1]]])

    ss = StandardScaler()
    X = npd.NWPFrame(ss.fit_transform(X), columns=X.keys())
    X = X.iloc[:vt]

    clfs = pickle.load(open(fitfile, 'rb'))[month_key_info[1]]

    p, c = predict(X, clfs, W[lclid][month_key_info[0]], smooth=False, confidence=True)

    vis_pred = adapt_visibility(p)
    vis = npd.NWPFrame(copy.deepcopy(df_date))
    vis['SKYNET-VIS'] = vis_pred
    # vis.rename(columns={'HEAD:YEAR': 'YEAR'}, inplace=True)
    c = pd.concat([copy.deepcopy(df_date), c], axis=1)
    # c.rename(columns={'HEAD:YEAR': 'YEAR'}, inplace=True)

    print(os.path.dirname(predfile))

    vis.to_csv(predfile, index=False)
    c.to_csv(conffile, index=False)
