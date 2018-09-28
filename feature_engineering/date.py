import numpy as np
import pandas as pd

from skynet.data_handling.base import match_keys


def grad_time(data, physical_quantity):
    features = list(data.keys())
    exts = match_keys(features, physical_quantity)
    grad = data[exts].diff(periods=1, axis=0)
    new_features = ["t_grad_%s" % f for f in exts]
    return pd.DataFrame(grad.values, columns=new_features)


def combine_date(data, header=None):
    if header is None:
        header = ["year", "month", "day", "hour", "min"]
    data[header] = data[header].astype(str)
    date = [y + mo.zfill(2) + d.zfill(2) + h.zfill(2) + mi.zfill(2) for y, mo, d, h, mi in data[header].values]
    data.drop(header, axis=1, inplace=True)
    data.insert(loc=0, column="date", value=date)
    return data


def convert_date(data, drop=False, method="sine"):
    if method == "sine":
        data = __sine_date(data)
    elif method == "split":
        data = __split_date(data)

    if drop:
        data.drop("date", axis=1, inplace=True)

    return data


def __sine_date(data):
    date = data["date"].astype(np.int).astype(np.str)
    hour = np.array([d[-4:-2] for d in date]).astype(np.int)
    hour = np.cos(2 * np.pi * hour / 24)
    data.insert(loc=0, column="hour", value=hour)

    years = np.unique([d[:4] for d in date])

    day = [0] * len(date)
    n = 0
    for year in years:
        if int(year) % 4 != 0:
            monthly_days = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            mdcum = np.cumsum(monthly_days)

        else:
            monthly_days = [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            mdcum = np.cumsum(monthly_days)

        ext = [d for d in date if d[:4] == year]
        day[n:n + len(day)] = [mdcum[int(d[4:6]) - 1] + int(d[6:8]) for d in ext]
        n += len(day)

    day = np.array(day)
    day = np.cos(2 * np.pi * day / 365)
    data.insert(loc=0, column="day", value=day)

    return data


def __split_date(data, replace=True):
    date = np.array([[int(d / 1e8), int(d / 1e6 % 1e2), int(d / 1e4 % 1e2), int(d / 1e2 % 1e2), int(d % 1e2)]
                     for d in data["date"]])
    if replace:
        data = data.drop("date", axis=1)
        for idx, d in enumerate(["year", "month", "day", "hour", "min"]):
            data.insert(loc=idx, column=d, value=date[:, idx])
    else:
        for idx, d in enumerate(["year", "month", "day", "hour", "min"]):
            data.insert(loc=idx + 1, column=d, value=date[:, idx])

    return data
