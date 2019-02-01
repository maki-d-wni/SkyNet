import os
import pickle
import shutil

import numpy as np
from pandas import DataFrame


class MSMBase(object):
    _params = {
        'surface': [
            'Pressure reduced to MSL',
            'u-component of wind',
            'v-component of wind',
            'Temperature',
            'Relative humidity',
            'Low cloud cover',
            'Medium cloud cover',
            'High cloud cover',
            'Total cloud cover',
            'Total precipitation'
        ],
        'upper': [
            'Geopotential height',
            'Relative humidity',
            'Temperature',
            'Vertical velocity [pressure]',
            'u-component of wind',
            'v-component of wind'
        ]
    }
    _level = {
        'surface': ['surface'],
        'upper': [
            '300',
            '400',
            '500',
            '600',
            '700',
            '800',
            '850',
            '900',
            '925',
            '950',
            '975',
            '1000'
        ]
    }
    _base_time = {
        'surface': ['%02d' % t for t in range(0, 24, 3)],
        'upper': ['%02d' % t for t in range(3, 24, 6)]
    }
    _validity_time = {
        'surface': ['%02d' % t for t in range(40)],
        'upper': ['%02d' % t for t in range(0, 40, 3)]
    }
    _shape = {
        'surface': (505, 481),
        'upper': (253, 241)
    }


class MSM(MSMBase):
    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, params):
        self._params = params

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, level):
        self._level = level

    @property
    def base_time(self):
        return self._base_time

    @base_time.setter
    def base_time(self, base_time):
        self._base_time = base_time

    @property
    def validity_time(self):
        return self._validity_time

    @validity_time.setter
    def validity_time(self, validity_time):
        self._validity_time = validity_time

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = shape

    def read(self, path):
        pass

    def show(self):
        pass


class DateBase(object):
    pass


class Date(DateBase):
    def __init__(self):
        super().__init__()

    def split_periodic(self):
        pass


def split_time_series(data, level="month", period=2):
    date = data["date"].astype(int).astype(str)
    data.index = date.values
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
            else:
                key = "month:%d-%d" % (idx, idx + period - 1)
                ms = ["{0:02d}".format(m) for m in range(idx, idx + period)]
                ext_date = [d for d in date if d[i:e] in ms]
                spd[key] = data.loc[ext_date].reset_index(drop=True)
    elif level == "day":
        # i = 6
        # e = 8
        raise NotImplementedError
    else:
        # i = 0
        # e = 8
        raise NotImplementedError

    data.reset_index(drop=True, inplace=True)

    return spd


class NWPFrame(DataFrame):
    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False, clone=True):
        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy)

        if clone:
            self._init_data = None
            self._set_init_data()

    @property
    def init_data(self):
        return self._init_data

    def _set_init_data(self):
        from copy import deepcopy
        self._init_data = deepcopy(self)

    def append(self, other, axis=0, key=None, ignore_index=False, verify_integrity=False, inplace=True, **kwargs):
        from pandas.core.reshape.concat import concat
        if axis == 0:
            new_data = concat([self, DataFrame(other)], axis=axis)
        else:
            new_data = concat([self, DataFrame(other)], axis=axis)

        if inplace:
            self._update_inplace(new_data)
        else:
            return new_data

    def sync(self, objs, sync_key, inplace=True):
        from copy import deepcopy
        from pandas.core.reshape.concat import concat

        def __set_sync_index():

            try:
                sync_index = new_data[sync_key].astype(int)
            except ValueError:
                sync_index = new_data[sync_key]

            try:
                sync_index = sync_index.astype(str)
            except ValueError:
                sync_index = new_data[sync_key]

            try:
                new_data.index = sync_index.values
            except ValueError:
                raise

            try:
                sync_index = objs[sync_key].astype(int)
            except ValueError:
                sync_index = objs[sync_key]

            try:
                sync_index = sync_index.astype(str)
            except ValueError:
                sync_index = objs[sync_key]

            try:
                objs.index = sync_index.values
            except ValueError:
                raise

        new_data = deepcopy(self)
        __set_sync_index()
        new_data = concat([new_data, objs], axis=1).reset_index(drop=True)

        if inplace:
            self._update_inplace(new_data)
        else:
            return new_data

    def query(self, expr, inplace=True, **kwargs):
        return super().query(expr, inplace=inplace, **kwargs)

    def eval(self, expr, inplace=True, **kwargs):
        return super().eval(expr, inplace=inplace, **kwargs)

    def ffill(self, axis=None, inplace=True, limit=None, downcast=None):
        return super().ffill(axis=axis, inplace=inplace, limit=limit, downcast=downcast)

    def bfill(self, axis=None, inplace=True, limit=None, downcast=None):
        return super().bfill(axis=axis, inplace=inplace, limit=limit, downcast=downcast)

    def fillna(self, value=None, method=None, axis=None, inplace=True,
               limit=None, downcast=None, **kwargs):
        return super().fillna(value=value, method=method, axis=axis, inplace=inplace,
                              limit=limit, downcast=downcast, **kwargs)

    def set_index(self, keys, drop=True, append=False, inplace=True,
                  verify_integrity=False):
        return super().set_index(keys, drop=drop, append=append, inplace=inplace,
                                 verify_integrity=verify_integrity)

    def reset_index(self, level=None, drop=False, inplace=True, col_level=0,
                    col_fill=''):
        return super().reset_index(level=level, drop=drop, inplace=inplace, col_level=col_level,
                                   col_fill=col_fill)

    def set_axis(self, labels, axis=0, inplace=None):
        return super().set_axis(labels, axis=axis, inplace=inplace)

    def rename_axis(self, mapper, axis=0, copy=True, inplace=True):
        return super().rename_axis(mapper, axis=axis, copy=copy, inplace=inplace)

    def drop(self, labels=None, axis=0, index=None, columns=None, level=None,
             inplace=True, errors='raise'):
        return super().drop(labels=labels, axis=axis, index=index, columns=columns, level=level,
                            inplace=inplace, errors=errors)

    def dropna(self, axis=0, how='any', thresh=None, subset=None,
               inplace=True):
        return super().dropna(axis=axis, how=how, thresh=thresh, subset=subset,
                              inplace=inplace)

    def drop_duplicates(self, subset=None, keep='first', inplace=True):
        return super().drop_duplicates(subset=subset, keep=keep, inplace=inplace)

    def sort_values(self, by, axis=0, ascending=True, inplace=True,
                    kind='quicksort', na_position='last'):
        return super().sort_values(by, axis=axis, ascending=ascending, inplace=inplace,
                                   kind=kind, na_position=na_position)

    def sort_index(self, axis=0, level=None, ascending=True, inplace=True,
                   kind='quicksort', na_position='last', sort_remaining=True,
                   by=None):
        return super().sort_index(axis=axis, level=level, ascending=ascending, inplace=inplace,
                                  kind=kind, na_position=na_position, sort_remaining=sort_remaining,
                                  by=by)

    def sortlevel(self, level=0, axis=0, ascending=True, inplace=True,
                  sort_remaining=True):
        return super().sortlevel(level=level, axis=axis, ascending=ascending, inplace=inplace,
                                 sort_remaining=sort_remaining)

    def consolidate(self, inplace=True):
        return super().consolidate(inplace=inplace)

    def replace(self, to_replace=None, value=None, inplace=True, limit=None,
                regex=False, method='pad', axis=None):
        return super().replace(to_replace=to_replace, value=value, inplace=inplace, limit=limit,
                               regex=regex, method=method, axis=axis)

    def interpolate(self, method='linear', axis=0, limit=None, inplace=True,
                    limit_direction='forward', downcast=None, **kwargs):
        return super().interpolate(method=method, axis=axis, limit=limit, inplace=inplace,
                                   limit_direction=limit_direction, downcast=downcast, **kwargs)

    def clip(self, lower=None, upper=None, axis=None, inplace=True,
             *args, **kwargs):
        return super().clip(lower=lower, upper=upper, axis=axis, inplace=inplace,
                            *args, **kwargs)

    def clip_upper(self, threshold, axis=None, inplace=True):
        return super().clip_upper(threshold, axis=axis, inplace=inplace)

    def clip_lower(self, threshold, axis=None, inplace=True):
        return super().clip_lower(threshold, axis=axis, inplace=inplace)

    def where(self, cond, other=np.nan, inplace=True, axis=None, level=None,
              errors='raise', try_cast=False, raise_on_error=None):
        return super().where(cond, other=other, inplace=inplace, axis=axis, level=level,
                             errors=errors, try_cast=try_cast, raise_on_error=raise_on_error)

    def mask(self, cond, other=np.nan, inplace=True, axis=None, level=None,
             errors='raise', try_cast=False, raise_on_error=None):
        return super().mask(cond, other=other, inplace=inplace, axis=axis, level=level,
                            errors=errors, try_cast=try_cast, raise_on_error=raise_on_error)


class MSMFrame(NWPFrame):
    pass


ndf = NWPFrame(
    [
        [1, 2, 3],
        ['a', 'b', 'c'],
        [4, 'd', 5]
    ],
    columns=['a', 'b', 'c']
)

ndf.sync(
    NWPFrame(
        [
            [6, 7],
            ['e', 'f'],
            ['g', 5]
        ],
        index=[1, 2, 3],
        columns=['a2', 'c']
    ),
    sync_key='c'
)

test = NWPFrame(
    [
        [1, 2, 3],
        [4, 5, 6]
    ],
    columns=['a', 'b', 'c']
)

print(ndf.init_data)
print(ndf)
for attr in dir(ndf):
    if attr == 'init_data':
        print('aaa')


def get_init_features():
    f = ['date', 'CAPE', 'CIN', 'SSI', 'WX_telop_100', 'WX_telop_200', 'WX_telop_300', 'WX_telop_340',
         'WX_telop_400', 'WX_telop_430', 'WX_telop_500', 'WX_telop_600', 'WX_telop_610',
         'Pressure reduced to MSL', 'Pressure', 'u-component of wind', 'v-component of wind', 'Temperature',
         'Relative humidity', 'Low cloud cover', 'Medium cloud cover', 'High cloud cover', 'Total cloud cover',
         'Total precipitation', 'Geop1000', 'u-co1000', 'v-co1000', 'Temp1000', 'Vert1000', 'Rela1000',
         'Geop975', 'u-co975', 'v-co975', 'Temp975', 'Vert975', 'Rela975', 'Geop950', 'u-co950', 'v-co950',
         'Temp950', 'Vert950', 'Rela950', 'Geop925', 'u-co925', 'v-co925', 'Temp925', 'Vert925', 'Rela925',
         'Geop900', 'u-co900', 'v-co900', 'Temp900', 'Vert900', 'Rela900', 'Geop850', 'u-co850', 'v-co850',
         'Temp850', 'Vert850', 'Rela850', 'Geop800', 'u-co800', 'v-co800', 'Temp800', 'Vert800', 'Rela800',
         'Geop700', 'u-co700', 'v-co700', 'Temp700', 'Vert700', 'Rela700', 'Geop600', 'u-co600', 'v-co600',
         'Temp600', 'Vert600', 'Rela600', 'Geop500', 'u-co500', 'v-co500', 'Temp500', 'Vert500', 'Rela500',
         'Geop400', 'u-co400', 'v-co400', 'Temp400', 'Vert400', 'Rela400', 'Geop300', 'u-co300', 'v-co300',
         'Temp300', 'Vert300', 'Rela300']
    return f


def get_init_response():
    r = ["visibility_rank"]
    return r


def get_init_vis_level():
    vis_level = {0: 0, 1: 800, 2: 1600, 3: 2600, 4: 3600, 5: 4800, 6: 6000, 7: 7400, 8: 8800}
    return vis_level


def read_learning_data(path):
    features = get_init_features()
    response = get_init_response()
    data = pickle.load(open(path, "rb"))
    data = data[features + response].reset_index(drop=True)
    return data


def split_binary(data, key):
    label = data[key]
    threshold = int(len(np.unique(label)) / 2)
    x1 = data[label <= threshold]
    x0 = data[label > threshold]

    x1.insert(loc=len(x1.keys()), column="binary", value=np.ones(len(x1)))
    x0.insert(loc=len(x0.keys()), column="binary", value=np.zeros(len(x0)))

    return x1, x0


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
