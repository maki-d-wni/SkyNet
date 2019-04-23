import os
import shutil
import copy
import datetime
import numpy as np
import pandas as pd


def majority_vote(x, n_class):
    mv = np.zeros((len(x), n_class))
    idx = np.arange(len(x))
    for i in range(x.shape[1]):
        mv[idx, x[:, i].astype(int)] += 1

    return mv.argmax(axis=1)


def weighted_majority_vote(x, w):
    xx = np.zeros_like(x)
    mp = np.zeros_like(x)
    ww = np.array([w for _ in range(len(x))])
    idx = np.arange(len(x))
    for i in range(x.shape[1]):
        mp[:, i] = ww[idx, x[:, i].astype(int)]
        xx[:, i] = x[:, i] * mp[:, i]

    return xx.sum(axis=1) / mp.sum(axis=1)


def smoothing(x, ksize):
    kernel = np.ones(ksize)
    x_sm = 1 / kernel.sum() * np.convolve(x, kernel, mode="same")

    extend = np.zeros_like(x)
    for i in range(len(x)):
        idx1 = i - int(ksize / 2)
        idx2 = i + int(ksize / 2)
        if x_sm[i] < 5.:
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


def read_visibility_human_edit(icao):
    he = pd.read_csv(
        "/Users/makino/PycharmProjects/SkyCC/data/csv/after/%s.csv" % icao,
        names=["a", "b", "date", "c", "VIS_after", "d", "e", "f", "g", "h", "i"]
    )
    he = he[["date", "VIS_after"]]
    he = he.drop_duplicates("date", keep="last")

    vr = convert_vis_rank(he["VIS_after"].values)
    he["visibility_rank"] = vr
    he.columns = ["date", "visibility", "visibility_rank"]

    return he


def make_vis_table(init_date: str, end_date: str, metar=None, af=None, he=None, ml=None):
    def __check_index(df):
        if "date" in df.keys():
            idx = df["date"].astype(int).astype(str).values
            df.index = idx
            df = df.drop("date", axis=1)
        return df

    def __append(df, values, kind):
        values = __check_index(values)
        for key in values:
            df["%s_%s" % (kind, key)] = values[key]

        return df

    ids = [init_date[0:4], init_date[4:6], init_date[6:8], init_date[8:10], init_date[10:12]]
    ids = [int(d) for d in ids]
    eds = [end_date[0:4], end_date[4:6], end_date[6:8], end_date[8:10], end_date[10:12]]
    eds = [int(d) for d in eds]

    s = datetime.datetime(ids[0], ids[1], ids[2], ids[3], ids[4])
    e = datetime.datetime(eds[0], eds[1], eds[2], eds[3], eds[4])

    days = (e - s).days

    date = [(s + datetime.timedelta(hours=i)).strftime("%Y%m%d%H%M") for i in range(0, 24 * days)]

    vis = pd.DataFrame(index=date)
    if metar is not None:
        vis = __append(vis, metar, kind="metar")
    if af is not None:
        vis = __append(vis, af, kind="AF")
    if he is not None:
        vis = __append(vis, he, kind="human")
    if ml is not None:
        vis = __append(vis, ml, kind="SkyNet")

    return vis


def convert_vis_rank(vis):
    from skynet.data_handling import get_init_vis_level

    label = np.zeros(len(vis))
    v = list(get_init_vis_level().values()) + [100000]
    delta = np.diff(v)
    for i, d in enumerate(delta):
        indices = np.where((vis > v[i]) & (vis <= v[i] + d))[0]
        label[indices] = i
    return label


def stack_daily_forecast(df, file_path):
    list_dir = os.listdir(file_path)
    list_dir.sort()
    for d in list_dir:
        if os.path.isdir(file_path + "/" + d):
            df = stack_daily_forecast(df, file_path + "/" + d)
        else:
            x = pd.read_html(file_path + "/" + d)[0]
            las = list(x.keys().labels)
            lvs = list(x.keys().levels)
            keys = [lvs[-1][l] for l in las[-1]]
            x.columns = keys

            df = pd.concat([df, x])

            return df

    return df


def run(icao, date, n_clfs, ws):
    import pickle
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import recall_score
    from skynet import OUTPUT_PATH
    from skynet.data_handling import get_init_response
    from skynet.data_handling import read_learning_data
    from skynet.data_handling import split_time_series
    from skynet.data_handling import extract_time_series
    from skynet.preprocessing import PreProcessor

    df = stack_daily_forecast(pd.DataFrame(),
                              file_path=os.getcwd() + "/archive")

    df = df.drop_duplicates("date", keep="last")
    df = df.reset_index(drop=True)
    print(df)

    preprocess = PreProcessor(norm=False, binary=False)
    test = read_learning_data(OUTPUT_PATH + "/datasets/apvis/test_%s.pkl" % icao)

    # feature増やしてからデータ構造を（入力、正解）に戻す
    preprocess.fit(X_test=test.iloc[:, :-1], y_test=test.iloc[:, -1])
    test = pd.concat([preprocess.X_test, preprocess.y_test], axis=1)

    # 評価対象月のデータを抽出
    period = 2
    keys = [
        "month:1-2",
        "month:3-4",
        "month:5-6",
        "month:7-8",
        "month:9-10",
        "month:11-12"
    ]
    month = datetime.datetime.now().month - 1
    if month % period == 0:
        kn = int(month / period) - 1
    else:
        kn = int(month / period)

    sptest = split_time_series(test, level="month", period=period)
    test = list(sptest.values())[kn]

    target = get_init_response()
    fets = [f for f in test.keys() if not (f in target)]

    # metar
    metar = test[["date", "visibility_rank"]]
    # human edit
    he = read_visibility_human_edit(icao)

    ss = StandardScaler()

    X_test = test[fets]
    init_date = str(int(X_test["date"].values[0]))
    end_date = str(int(X_test["date"].values[-1]))

    X_test = pd.DataFrame(ss.fit_transform(X_test), columns=X_test.keys())
    y_test = sptest[key][target]

    y_true = np.where(y_test > 1, 0, 1)
    p_rf = np.zeros((len(X_test), n_clfs[i_term]))
    score_rf = np.zeros(n_clfs[i_term])

    learning_model = True
    if learning_model:
        for i in range(n_clfs[i_term]):
            clf = pickle.load(open(OUTPUT_PATH + "/learning_models/%s/forest/%s/%s/rf%03d.pkl"
                                   % (icao, date, key, i), "rb"))
            p_rf[:, i] = clf.predict(X_test.values)

            y_pred = np.where(p_rf[:, i] > 1, 0, 1)
            score_rf[i] = recall_score(y_true=y_true, y_pred=y_pred)

    p = p_rf.mean(axis=1)
    score_rf = score_rf.mean()
    print("f1 mean", score_rf)

    # p = majority_vote(p_rf, n_class=9)
    p = weighted_majority_vote(p_rf, ws[i_term])
    p = smoothing(p, ksize=12)

    ml = copy.deepcopy(sptest[key][["date"]])
    ml["visibility_rank"] = p

    vis = make_vis_table(init_date, end_date, metar=metar, he=he, ml=ml)

    plt.plot(vis["metar_visibility_rank"].values)
    # plt.plot(vis["human_visibility_rank"].values)
    plt.plot(vis["SkyNet_visibility_rank"].values)
    plt.show()


def main():
    icao = "RJFK"
    date = "20180827"
    n_clfs = [
        100,
        100,
        100,
        100,
        100,
        100
    ]

    ws = {
        "RJFK": [
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [3, 1, 1, 1, 1, 1, 1, 1, 1],
            [3, 1, 1, 1, 1, 1, 1, 1, 1],
            [10, 1, 1, 1, 1, 1, 1, 1, 1],
            [10, 1, 1, 1, 1, 1, 1, 1, 1],
            [30, 1, 1, 1, 1, 1, 1, 1, 1]
        ],
        "RJFT": [
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [3, 1, 1, 1, 1, 1, 1, 1, 1],
            [10, 1, 1, 1, 1, 1, 1, 1, 1],
            [10, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1]
        ]
    }

    run(icao, date, n_clfs, ws[icao])


if __name__ == "__main__":
    main()
