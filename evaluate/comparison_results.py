import os
import datetime
import numpy as np
import pandas as pd


def set_visibility_metar(icao):
    from skynet import DATA_PATH
    from skynet.data_handling import strtime_to_datetime
    metar = pd.read_csv(
        DATA_PATH + "/csv/metar/airport_vis/metar_%s.csv" % icao
    )
    metar["visibility_rank"] = convert_visibility_rank(metar["visibility"].values)
    metar.index = strtime_to_datetime(metar["date"].values)
    return metar


def set_visibility_human_edit(icao):
    from skynet import DATA_PATH
    from skynet.data_handling import strtime_to_datetime
    he = pd.read_csv(
        DATA_PATH + "/csv/after/%s.csv" % icao,
        names=["a", "b", "date", "c", "VIS_after", "d", "e", "f", "g", "h", "i"]
    )
    he = he[["date", "VIS_after"]]
    he = he.drop_duplicates("date", keep="last")

    vr = convert_visibility_rank(he["VIS_after"].values)
    he["visibility_rank"] = vr
    he.index = strtime_to_datetime(he["date"].values)
    he.columns = ["date", "visibility", "visibility_rank"]

    return he


def set_visibility_area_forecast(icao):
    import pickle
    from skynet import DATA_PATH
    from skynet.data_handling import strtime_to_datetime
    af = pickle.load(open(DATA_PATH + "/pickle/learning/skynet/test_%s.pkl" % icao, "rb"))
    af = af[["date", "VIS"]].rename(columns={"VIS": "visibility"})
    af["visibility_rank"] = convert_visibility_rank(af["visibility"].values)
    af.index = strtime_to_datetime(af["date"].values)

    return af


def convert_visibility_rank(vis):
    from skynet.data_handling import get_init_vis_level

    label = np.zeros(len(vis))
    v = list(get_init_vis_level().values()) + [100000]
    delta = np.diff(v)
    for i, d in enumerate(delta):
        indices = np.where((vis > v[i]) & (vis <= v[i] + d))[0]
        label[indices] = i
    return label


def f1_recall_precision_score(y_true, y_pred, threshold=None):
    from sklearn.metrics import f1_score, recall_score, precision_score

    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    y_true = np.where(y_true >= threshold, 0, 1)
    y_pred = np.where(y_pred >= threshold, 0, 1)
    f1 = f1_score(y_true, y_pred)
    rs = recall_score(y_true, y_pred)
    ps = precision_score(y_true, y_pred)

    return f1, rs, ps


def __print_specified_term_result(string, pf1, prs, pps, length=12):
    space = [" " for _ in range(length - len(string))]
    for s in space:
        string += s
    print("%s: f1 = %0.3f, recall = %0.3f, precision = %0.3f" % (string, pf1, prs, pps))


def main():
    icao = "RJCC"
    cpr = pd.read_csv(os.getcwd() + "/comparison/forecast_data_%s_2017.csv" % icao)
    date = cpr["datetime"].values
    date = [datetime.datetime.strptime(d[:16], "%Y-%m-%d %H:%M") for d in date]
    cpr.index = date
    vis = set_visibility_metar(icao)
    vis["800"] = cpr["800"]

    af = set_visibility_area_forecast(icao)
    he = set_visibility_human_edit(icao)

    vis["AF"] = af["visibility_rank"]
    vis["human"] = he["visibility_rank"]
    vis = vis.dropna()
    print(vis)

    y_true = vis["visibility_rank"].values
    y_pred = np.where(vis["800"], 0, 8)

    f1, rs, ps = f1_recall_precision_score(y_true=y_true, y_pred=y_pred, threshold=1)
    __print_specified_term_result("score", f1, rs, ps)


if __name__ == "__main__":
    main()
