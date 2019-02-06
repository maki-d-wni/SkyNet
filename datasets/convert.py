import datetime
import numpy as np
import pandas as pd


def split_blocks(X, y, n_folds=3):
    idx = {int(l): np.where(y.values[:, 0] == l)[0] for l in np.unique(y.values[:, 0])}
    spidx = {}
    for i in range(n_folds):
        spidx[i] = {
            int(l): idx[l][i * int(len(idx[l]) / n_folds):(i + 1) * int(len(idx[l]) / n_folds)]
            for l in idx
        }
        cnc = []
        for k in spidx[i]:
            cnc += list(spidx[i][k])
        spidx[i] = cnc

    spX = {i: X.loc[spidx[i]] for i in range(n_folds)}

    if type(y) != pd.DataFrame:
        y = pd.DataFrame(y)
    spy = {i: y.loc[spidx[i]] for i in range(n_folds)}

    return spX, spy


def split_binary(data, key):
    label = data[key]
    threshold = int(len(np.unique(label)) / 2)
    x1 = data[label <= threshold]
    x0 = data[label > threshold]

    x1.insert(loc=len(x1.keys()), column="binary", value=np.ones(len(x1)))
    x0.insert(loc=len(x0.keys()), column="binary", value=np.zeros(len(x0)))

    return x1, x0


def split_time_series(data, date, level="month", period=2, index_date=False):
    date = date.astype(int).astype(str)
    data.index = date
    spd = {}
    if level == "year":
        # i = 0
        # e = 4
        raise NotImplementedError
    elif level == "month":
        i = 4
        e = 6
        for idx in range(1, 13, period):
            if idx + period > 12:
                key = "month:%d-%d" % (idx, 12)
                ms = ["{0:02d}".format(m) for m in range(idx, 13)]
                ext_date = [d for d in date if d[i:e] in ms]
                spd[key] = data.loc[ext_date].reset_index(drop=True)
                if index_date:
                    spd[key].index = _strtime_to_datetime(ext_date)
            else:
                key = "month:%d-%d" % (idx, idx + period - 1)
                ms = ["{0:02d}".format(m) for m in range(idx, idx + period)]
                ext_date = [d for d in date if d[i:e] in ms]
                spd[key] = data.loc[ext_date].reset_index(drop=True)
                if index_date:
                    spd[key].index = _strtime_to_datetime(ext_date)
    elif level == "day":
        # i = 6
        # e = 8
        raise NotImplementedError
    else:
        # i = 0
        # e = 8
        raise NotImplementedError

    return spd


def _strtime_to_datetime(date):
    date = np.array(date).astype(int).astype(str)
    date = [datetime.datetime.strptime(d, "%Y%m%d%H%M") for d in date]
    return date


def _datetime_to_strtime(date):
    date = [d.strftime("%Y%m%d%H%M") for d in date]
    return date


def balanced(X, y):
    if type(y) == pd.DataFrame:
        y = y.values[:, 0]

    indices = [np.where(y == l)[0] for l in np.unique(y)]
    max_vis_indices = indices[-1]
    len_indices = [len(idx) for idx in indices[:-1]]
    len_indices = max(len_indices)
    indices[-1] = np.random.choice(max_vis_indices, len_indices)

    new_indices = []
    for idx in indices:
        new_indices += list(idx)

    shuffled = np.random.choice(new_indices, len(new_indices), replace=False)

    if type(X) == pd.DataFrame:
        X = X.iloc[shuffled].reset_index(drop=True)
        y = pd.DataFrame(y)
        y = y.iloc[shuffled].reset_index(drop=True)
    elif type(X) == np.ndarray:
        X = X[shuffled]
        y = y[shuffled]

    return X, y
