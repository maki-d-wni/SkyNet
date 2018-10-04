import copy
import pickle
import numpy as np
import pandas as pd
import skynet.data_handling as dh

from skynet import OUTPUT_PATH, HTML_PATH

N_CLF = 100

W = {
    "RJFK": [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [5, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [5, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 5]
    ],
    "RJFT": [
        [1, 1, 1, 1, 1, 1, 1, 1, 10],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [50, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 50],
        [10, 1, 1, 1, 1, 1, 1, 1, 1]
    ],
    "RJCC": [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]
}


def predict(X, clfs, w, smooth=False):
    n_clf = len(clfs)
    p_rf = np.zeros((len(X), n_clf))

    for i, clf in enumerate(clfs):
        p_rf[:, i] = clf.predict(X.values)

    # p = p_rf.mean(axis=1)
    # p = majority_vote(p_rf, n_class=9, threshold=None)
    p = deflected_mean(p_rf, w, threshold=6)
    if smooth:
        p = smoothing(p, ksize=6, threshold=5)

    return p


def deflected_mean(x, w, threshold=None):
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


def set_month_key(date, period=2):
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
    v = copy.deepcopy(v)
    vis_level = dh.get_init_vis_level()
    diff = np.diff(list(vis_level.values()) + [9999])
    for key in vis_level:
        v[(v > key) & (v <= (key + 1))] = \
            diff[key] * (v[(v > key) & (v <= (key + 1))] - key) + vis_level[key]
    v[v >= list(vis_level.values())[-1]] = 9999
    return v


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--icao")
    parser.add_argument("--date")
    parser.add_argument("--time")

    args = parser.parse_args()

    icao = args.icao
    date = args.date
    time = args.time

    if args.icao is None:
        icao = "RJCC"

    if args.date is None:
        date = "20180620"

    if args.time is None:
        time = "060000"

    X = pd.read_csv("%s/live/input/norm_%s.csv" % (OUTPUT_PATH, icao))

    key = set_month_key(date)
    clfs = [
        pickle.load(
            open("%s/learning_models/%s/forest/%s/rf%03d.pkl" % (OUTPUT_PATH, icao, key[1], i), "rb"))
        for i in range(N_CLF)
    ]

    p = predict(X, clfs, W[icao][key[0]], smooth=False)

    vis = pd.read_csv("%s/live/input/%s.csv" % (OUTPUT_PATH, icao))[["date"]]
    vis_pred = adapt_visibility(p)
    vis["SkyNet"] = vis_pred
    vis.to_csv("%s/live/prediction/%s.csv" % (OUTPUT_PATH, icao), index=False)


if __name__ == "__main__":
    main()
