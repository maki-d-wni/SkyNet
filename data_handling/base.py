import os
import shutil
import pickle
import datetime
import numpy as np
import pandas as pd

from skynet import DATA_PATH


def get_init_features():
    f = [
        'date',
        'CAPE', 'CIN', 'SSI', 'WX_telop_100', 'WX_telop_200', 'WX_telop_300', 'WX_telop_340',
        'WX_telop_400', 'WX_telop_430', 'WX_telop_500', 'WX_telop_600', 'WX_telop_610',
        'Pressure reduced to MSL',
        'Pressure',
        'u-component of wind',
        'v-component of wind',
        'Temperature',
        'Relative humidity',
        'Low cloud cover',
        'Medium cloud cover',
        'High cloud cover',
        'Total cloud cover',
        'Total precipitation',
        'Geop1000',
        'u-co1000',
        'v-co1000',
        'Temp1000',
        'Vert1000',
        'Rela1000',
        'Geop975',
        'u-co975',
        'v-co975',
        'Temp975',
        'Vert975',
        'Rela975',
        'Geop950',
        'u-co950',
        'v-co950',
        'Temp950',
        'Vert950',
        'Rela950',
        'Geop925',
        'u-co925',
        'v-co925',
        'Temp925',
        'Vert925',
        'Rela925',
        'Geop900',
        'u-co900',
        'v-co900',
        'Temp900',
        'Vert900',
        'Rela900',
        'Geop850',
        'u-co850',
        'v-co850',
        'Temp850',
        'Vert850',
        'Rela850',
        'Geop800',
        'u-co800',
        'v-co800',
        'Temp800',
        'Vert800',
        'Rela800',
        'Geop700',
        'u-co700',
        'v-co700',
        'Temp700',
        'Vert700',
        'Rela700',
        'Geop600',
        'u-co600',
        'v-co600',
        'Temp600',
        'Vert600',
        'Rela600',
        'Geop500',
        'u-co500',
        'v-co500',
        'Temp500',
        'Vert500',
        'Rela500',
        'Geop400',
        'u-co400',
        'v-co400',
        'Temp400',
        'Vert400',
        'Rela400',
        'Geop300',
        'u-co300',
        'v-co300',
        'Temp300',
        'Vert300',
        'Rela300'
    ]
    return f


def get_init_response():
    r = ["visibility_rank"]
    return r


def get_init_vis_level():
    vis_level = {0: 0, 1: 800, 2: 1600, 3: 2600, 4: 3600, 5: 4800, 6: 6000, 7: 7400, 8: 8800}
    return vis_level


def concat(data, value):
    new_feature = list(value.keys())
    for nf in new_feature:
        ins_idx = len(data.keys())
        data.insert(loc=ins_idx, column=nf, value=value[nf])
    return data


def drop(data, drop_keys=()):
    return data.drop(drop_keys, axis=1, inplace=False)


def read_learning_data(path):
    features = get_init_features()
    response = get_init_response()
    data = pickle.load(open(path, "rb"))
    data = data[features + response].reset_index(drop=True)
    return data


def sync_values(base, x, key):
    base.index = base[key].astype(int).astype(str).values
    x.index = x[key].astype(int).astype(str).values

    for k in x:
        base[k] = x[k]

    base = base.reset_index(drop=True)
    base = base.dropna()

    return base


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
                    spd[key].index = strtime_to_datetime(ext_date)
            else:
                key = "month:%d-%d" % (idx, idx + period - 1)
                ms = ["{0:02d}".format(m) for m in range(idx, idx + period)]
                ext_date = [d for d in date if d[i:e] in ms]
                spd[key] = data.loc[ext_date].reset_index(drop=True)
                if index_date:
                    spd[key].index = strtime_to_datetime(ext_date)
    elif level == "day":
        # i = 6
        # e = 8
        raise NotImplementedError
    else:
        # i = 0
        # e = 8
        raise NotImplementedError

    return spd


def strtime_to_datetime(date):
    date = np.array(date).astype(int).astype(str)
    date = [datetime.datetime.strptime(d, "%Y%m%d%H%M") for d in date]
    return date


def datetime_to_strtime(date):
    date = [d.strftime("%Y%m%d%H%M") for d in date]
    return date


def extract_time_series(data, level="month", init=1, end=2):
    date = data["date"].astype(int).astype(str)
    data.index = date.values
    if level == "year":
        # i = 0
        # e = 4
        raise NotImplementedError
    elif level == "month":
        i = 4
        e = 6

        ms = ["{0:02d}".format(m) for m in range(init, end + 1)]
        print(ms)
        ext_date = [d for d in date if d[i:e] in ms]
        data = data.loc[ext_date].reset_index(drop=True)

    elif level == "day":
        # i = 6
        # e = 8
        raise NotImplementedError
    else:
        # i = 0
        # e = 8
        raise NotImplementedError

    return data


def match_keys(features, extracted, init=0, end=4):
    return [f for f in features if f[init:end] == extracted]


def match_keys_index(features, extracted, init=0, end=4):
    exts = [f for f in features if f[init:end] == extracted]
    idx = [features.index(f) for f in exts]
    return idx


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


def file_arrangement(file_path, output_path, tag_id):
    list_dir = os.listdir(file_path)
    list_dir.sort()
    for d in list_dir:
        if os.path.isdir(file_path + "/" + d):
            file_arrangement(file_path + "/" + d, output_path, tag_id)
        else:
            if os.path.isfile(DATA_PATH + "/%s/" % tag_id + d):
                print(d, "already exist.")
            else:
                print(d)
                shutil.copyfile(file_path + "/" + d, output_path + "/" + d)
    return


def convert_dict_construction(old, new: dict, pwd: str, depth: int):
    new = __apply_convert_dict_construction(old, new, pwd)
    keys = list(new.keys())
    skeys = [key.split(pwd) for key in keys]
    tn = []
    for i, d in enumerate(skeys):
        if len(d) > 2:
            tn.append(len(d) - 1 - depth)
        else:
            tn.append(1)

    keys = ["/".join(key[d:]) for d, key in zip(tn, skeys)]

    new = {key: new[n] for key, n in zip(keys, new)}

    return new


def __apply_convert_dict_construction(old, new: dict, pwd: str):
    if type(old) == dict:
        for o in old:
            nkey = pwd
            if type(o) == str:
                if pwd == "/":
                    nkey += o
                else:
                    nkey += "/" + o
            if __check_iterable(old[o]):
                __apply_convert_dict_construction(old[o], new, nkey)
            else:
                if nkey in new.keys():
                    new[nkey].append(old[o])
                else:
                    new[nkey] = [old[o]]
    else:
        for o in old:
            nkey = pwd
            if type(o) == str:
                if pwd == "/":
                    nkey += o
                else:
                    nkey += "/" + o
            if __check_iterable(o):
                __apply_convert_dict_construction(o, new, nkey)

    return new


def __check_iterable(obj):
    if hasattr(obj, "__iter__"):
        if type(obj) == str:
            return False
        else:
            return True
    else:
        return False
