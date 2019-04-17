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


def split_binary(X, y, threshold=None):
    if threshold is None:
        ma = y.max()
        mi = y.min()
        threshold = int((ma - mi) / 2)

    x1 = X[y <= threshold]
    x0 = X[y > threshold]

    x1.insert(loc=len(x1.keys()), column="binary", value=np.ones(len(x1)))
    x0.insert(loc=len(x0.keys()), column="binary", value=np.zeros(len(x0)))

    return x1, x0


def split_time_series(X, date, date_fmt='%Y-%m-%d %H:%M', level="month", period=2):
    xtype = type(X)
    if xtype == np.ndarray:
        if X.ndim == 1:
            X = pd.Series(X)
        else:
            X = pd.DataFrame(X)

    try:
        date = date.astype(int)
    except ValueError:
        pass
    try:
        date = date.astype(str)
    except ValueError:
        pass

    spd = {}
    if level == "year":
        raise NotImplementedError
    elif level == "month":
        date = np.array([datetime.datetime.strptime(d, date_fmt).month for d in date])
        for idx in range(1, 13, period):
            if idx + period > 12:
                key = "month:%d-%d" % (idx, 12)
                ms = [m for m in range(idx, 13)]
                spd[key] = X.iloc[(date == ms[0]) | (date == ms[1])].reset_index(drop=True)
            else:
                key = "month:%d-%d" % (idx, idx + period - 1)
                ms = [m for m in range(idx, idx + period)]
                spd[key] = X.iloc[(date == ms[0]) | (date == ms[1])].reset_index(drop=True)
    elif level == "day":
        raise NotImplementedError
    else:
        raise NotImplementedError

    if xtype == np.ndarray:
        return {key: spd[key].values for key in spd}
    else:
        return spd


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


def __type_check(obj):
    if type(obj) == pd.DataFrame:
        return obj.vlaues
    elif type(obj) == np.ndarray:
        return obj
    else:
        return


def main():
    from skynet import DATA_DIR
    from skynet.nwp2d import NWPFrame

    icao = 'RJOT'

    # metar読み込み
    metar_path = '%s/text/metar' % DATA_DIR
    with open('%s/head.txt' % metar_path, 'r') as f:
        header = f.read()
    header = header.split(sep=',')

    data15 = pd.read_csv('%s/2015/%s.txt' % (metar_path, icao), sep=',')
    data16 = pd.read_csv('%s/2016/%s.txt' % (metar_path, icao), sep=',')
    data17 = pd.read_csv('%s/2017/%s.txt' % (metar_path, icao), sep=',', names=header)

    metar_data = pd.concat([data15, data16, data17])
    metar_data = NWPFrame(metar_data)

    metar_data.strtime_to_datetime('date', '%Y%m%d%H%M%S', inplace=True)
    metar_data.datetime_to_strtime('date', '%Y-%m-%d %H:%M', inplace=True)
    metar_data.drop_duplicates('date', inplace=True)
    metar_data.index = metar_data['date'].values

    metar_keys = ['date', 'visibility', 'str_cloud']
    metar_data = metar_data[metar_keys]

    # MSM読み込み
    msm_data = pd.read_csv('%s/msm_airport/%s.csv' % (DATA_DIR, icao))

    msm_data.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
    msm_data.index = msm_data['date'].values
    msm_data.sort_index(inplace=True)

    X = pd.concat([msm_data, metar_data], axis=1)

    x0, x1 = split_binary(X, X['visibility'])
    print(x0)
    print()
    print(x1)


if __name__ == '__main__':
    main()
