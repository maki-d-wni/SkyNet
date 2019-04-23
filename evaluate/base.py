import os
import copy
import datetime
import numpy as np
import pandas as pd

N_CLF = [
    100,
    100,
    100,
    100,
    100,
    100
]

"""
W = {
    "RJFK": [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [5, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [5, 1, 1, 1, 1, 1, 1, 1, 1],
        [5, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1]
    ],
    "RJFT": [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [5, 1, 1, 1, 1, 1, 1, 1, 1],
        [30, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [5, 1, 1, 1, 1, 1, 1, 1, 1]
    ],
    "RJCC": [
        [5, 1, 1, 1, 1, 1, 1, 1, 1],
        [5, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [20, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [5, 1, 1, 1, 1, 1, 1, 1, 1]
    ]
}
"""

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
        [5, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [20, 1, 1, 1, 1, 1, 1, 1, 5],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 5]
    ]
}


def majority_vote(x, n_class, threshold=None):
    mv = np.zeros((len(x), n_class))
    idx = np.arange(len(x))
    for i in range(x.shape[1]):
        mv[idx, x[:, i].astype(int)] += 1

    p = mv.argmax(axis=1)

    if threshold is not None:
        p[p > threshold] = p.max()

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


def set_visibility_human_edit(icao):
    from skynet import DATA_PATH
    he = pd.read_csv(
        DATA_PATH + "/csv/after/%s.csv" % icao,
        names=["a", "b", "date", "c", "VIS_after", "d", "e", "f", "g", "h", "i"]
    )
    he = he[["date", "VIS_after"]]
    he = he.drop_duplicates("date", keep="last")

    vr = convert_visibility_rank(he["VIS_after"].values)
    he["visibility_rank"] = vr
    he.columns = ["date", "visibility", "visibility_rank"]

    return he


def set_visibility_metar(icao):
    from skynet import DATA_PATH
    metar = pd.read_csv(
        DATA_PATH + "/csv/metar/airport_vis/metar_%s.csv" % icao
    )
    return metar


def sync_values(base, x, key):
    base.index = base[key].astype(int).astype(str).values
    x.index = x[key].astype(int).astype(str).values

    for k in x:
        base[k] = x[k]

    base = base.reset_index(drop=True)

    return base


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


def make_vis_table_by_period(metar, he, ml, nodal):
    spvis = {}

    for key in metar:
        init_date = str(int(metar[key]["date"].values[0]))
        end_date = str(int(metar[key]["date"].values[-1]))

        vis = make_vis_table(
            init_date, end_date,
            metar=metar[key],
            he=he[key],
            ml=ml[key]
        )
        vis = vis.dropna()
        vis = convert_vis_level(
            vis,
            nodal=nodal,
            metar=metar[key],
            he=he[key],
            ml=ml[key]
        )

        spvis[key] = vis

    return spvis


def convert_visibility_rank(vis):
    from skynet.data_handling import get_init_vis_level

    label = np.zeros(len(vis))
    v = list(get_init_vis_level().values()) + [100000]
    delta = np.diff(v)
    for i, d in enumerate(delta):
        indices = np.where((vis > v[i]) & (vis <= v[i] + d))[0]
        label[indices] = i
    return label


def adapt_visibility(v, v_min, v_max):
    return 9999 * (v - v_min) / (v_max - v_min)


def convert_vis_level(data, nodal, metar=None, af=None, he=None, ml=None):
    def __apply(v):
        r = np.zeros_like(v)
        for i, d, in enumerate(delta):
            r[(v > nodal[i]) & (v <= nodal[i] + d)] = i

        return r.astype(int)

    def __append(df, values, kind):
        index = values["date"].astype(int).astype(str)
        r = pd.DataFrame(__apply(values["visibility"].values),
                         index=index,
                         columns=["new_%s_visibility_rank" % kind])
        df["new_%s_visibility_rank" % kind] = r["new_%s_visibility_rank" % kind]
        return df

    delta = np.diff(nodal)
    if metar is not None:
        data = __append(data, metar, kind="metar")
    if af is not None:
        data = __append(data, af, kind="AF")
    if he is not None:
        data = __append(data, he, kind="human")
    if ml is not None:
        data = __append(data, ml, kind="SkyNet")

    return data


def f1_recall_precision_score(y_true, y_pred, threshold=None):
    from sklearn.metrics import f1_score, recall_score, precision_score

    y_true = np.where(y_true >= threshold, 0, 1)
    y_pred = np.where(y_pred >= threshold, 0, 1)
    f1 = f1_score(y_true, y_pred)
    rs = recall_score(y_true, y_pred)
    ps = precision_score(y_true, y_pred)

    return f1, rs, ps


def extend_conf_mat(y_true, y_pred, rank, threats=None, true_name=None, pred_name=None):
    cfm = np.zeros((rank, rank))
    for i in range(rank):
        for j in range(rank):
            cfm[i, j] = len(np.where((y_true == i) & (y_pred == j))[0])
    if threats is None:
        y1 = [str(i) for i in range(rank)]
        y2 = [str(i) for i in range(1, rank + 1)]
        y2[-1] = ""
        threats = [[i, j] for i, j in zip(y1, y2)]
    else:
        y1 = threats
        y2 = [t for t in threats if t != y1[0]] + [""]
        threats = [[i, j] for i, j in zip(y1, y2)]

    if true_name is None:
        true_name = "Act"
    if pred_name is None:
        pred_name = "Pred"

    cfm = pd.DataFrame(
        cfm.astype(int),
        index=[
            [true_name for _ in range(rank)],
            [i.center(4, " ") + "-".center(3, " ") + j.center(4, " ") for i, j in threats]
        ],
        columns=[
            [pred_name for _ in range(rank)],
            [i.center(4, " ") + "-".center(3, " ") + j.center(4, " ") for i, j in threats]
        ]
    )

    return cfm


def make_conf_mat(vis, rank, threats, metar_vs_human=True, metar_vs_ml=True, human_vs_ml=True):
    if metar_vs_human:
        mat_he = extend_conf_mat(
            y_true=vis["new_metar_visibility_rank"],
            y_pred=vis["new_human_visibility_rank"],
            rank=rank,
            threats=threats,
            true_name="metar",
            pred_name="human",
        )
    else:
        mat_he = None

    if metar_vs_ml:
        mat_ml = extend_conf_mat(
            y_true=vis["new_metar_visibility_rank"],
            y_pred=vis["new_SkyNet_visibility_rank"],
            rank=rank,
            threats=threats,
            true_name="metar",
            pred_name="ML",
        )
    else:
        mat_ml = None

    if human_vs_ml:
        mat_heml = extend_conf_mat(
            y_true=vis["new_human_visibility_rank"],
            y_pred=vis["new_SkyNet_visibility_rank"],
            rank=rank,
            threats=threats,
            true_name="human",
            pred_name="ML",
        )
    else:
        mat_heml = None

    return mat_he, mat_ml, mat_heml


def make_conf_mat_by_period(metar, he, ml, rank, nodal):
    cfm_he = {}
    cfm_ml = {}
    cfm_heml = {}

    threats = np.array(nodal)[:-1].astype(str)

    for key in metar:
        init_date = str(int(metar[key]["date"].values[0]))
        end_date = str(int(metar[key]["date"].values[-1]))

        vis = make_vis_table(
            init_date, end_date,
            metar=metar[key],
            he=he[key],
            ml=ml[key]
        )
        vis = vis.dropna()
        vis = convert_vis_level(
            vis,
            nodal=nodal,
            metar=metar[key],
            he=he[key],
            ml=ml[key]
        )

        cfm_he[key], cfm_ml[key], cfm_heml[key] = make_conf_mat(vis, rank=rank, threats=threats)

    return cfm_he, cfm_ml, cfm_heml


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


def predict_by_period(X, icao, smooth=False):
    import pickle
    from sklearn.preprocessing import StandardScaler
    from skynet import MODEL_PATH
    from skynet.data_handling import get_init_response

    def __print_specified_term_result(string, pf1, prs, pps, length=12):
        space = [" " for _ in range(length - len(string))]
        for s in space:
            string += s
        print("%s: f1 = %0.3f, recall = %0.3f, precision = %0.3f" % (string, pf1, prs, pps))

    pred = {}
    f1t = 0
    rst = 0
    pst = 0
    for i_term, key in enumerate(X):
        ss = StandardScaler()
        x = X[key]
        target = get_init_response()
        fets = [f for f in x.keys() if not (f in target)]

        x = x[fets]
        x = pd.DataFrame(ss.fit_transform(x), columns=x.keys())
        y = X[key][target]

        # モデルを用意
        clfs = [
            pickle.load(
                open(MODEL_PATH + "/%s/forest/%s/rf%03d.pkl" % (icao, key, i), "rb"))
            for i in range(N_CLF[i_term])
        ]

        p = predict(x, clfs, W[icao][i_term], smooth)
        pred[key] = copy.deepcopy(X[key][["date"]])
        pred[key]["visibility"] = adapt_visibility(p, 0, 8)
        pred[key]["visibility_rank"] = p
        f1, rs, ps = f1_recall_precision_score(y, p, threshold=1)
        f1t += f1
        rst += rs
        pst += ps
        __print_specified_term_result(key, f1, rs, ps)

    __print_specified_term_result("total", f1t / len(X), rst / len(X), pst / len(X))

    return pred


def make_animation(vis):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    path = os.getcwd() + "/movie"
    os.makedirs(path, exist_ok=True)

    date = vis.index.values

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    imgs = []
    for i in range(100):
        if i > 50:
            img = ax.plot(vis["metar_visibility"][i - 50:i].values, c="b", label="metar")
            img += ax.plot(vis["human_visibility"][i - 50:i].values, c="g", label="human")
            # img += ax.plot(vis["SkyNet_visibility"][i - 50:i].values, c="r", label="SkyNet")
            img += ax.legend(loc="lower right", fontsize="14")
            plt.xticks([])
            plt.title("metar vs human")
        else:
            img = ax.plot(vis["metar_visibility"][:i].values, c="b", label="metar")
            img += ax.plot(vis["human_visibility"][:i].values, c="g", label="human")
            # img += ax.plot(vis["SkyNet_visibility"][:i].values, c="r",label="SkyNet")
            plt.xticks([])
            plt.title("metar vs human")
        imgs.append(img)
        # imgs.append(f3)

    ani = animation.ArtistAnimation(fig, imgs, interval=200)
    ani.save(path + "/vis_human.mp4", writer="ffmpeg")

    plt.show()


def make_animation2(vis, act="metar", pred="SkyNet", act_color="b", pred_color="g",
                    ani_name="vis_mehe.mp4"):
    import datetime
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    path = os.getcwd() + "/movie"
    os.makedirs(path, exist_ok=True)
    date = vis.index.values

    date = np.array(
        [str(datetime.datetime(int(d[:4]), int(d[4:6]), int(d[6:8]), int(d[8:10]), int(d[10:12])))
         for d in date]
    )

    fig = plt.figure(figsize=(8, 6))
    fig.add_subplot()
    fig.subplots_adjust(top=0.9, bottom=0.25)

    def update(i):
        s = 200
        if i != 0:
            plt.cla()

        if i > s:
            plt.plot(vis["%s_visibility" % act][i - s:i].values, c=act_color, label=act)
            plt.plot(vis["%s_visibility" % pred][i - s:i].values, c=pred_color, label=pred)
            plt.legend(loc="lower right", fontsize="14")
            plt.xticks(np.arange(s)[::20], date[i - s:i:20], rotation=45)
            plt.title("%s vs %s" % (act, pred), fontsize="14")
            plt.ylim(0, 10500)

        elif i != 0 and i <= s:
            plt.plot(vis["%s_visibility" % act][:i].values, c=act_color, label=act)
            plt.plot(vis["%s_visibility" % pred][:i].values, c=pred_color, label=pred)
            plt.legend(loc="lower right", fontsize="14")
            plt.xticks(np.arange(s)[::20], date[:s:20], rotation=45)
            plt.title("%s vs %s" % (act, pred), fontsize="14")
            plt.ylim(0, 10500)

        if i == len(vis):
            plt.plot(vis["%s_visibility" % act].values, c=act_color, label=act)
            plt.plot(vis["%s_visibility" % pred].values, c=pred_color, label=pred)
            plt.legend(loc="lower right", fontsize="14")
            plt.xticks(np.arange(len(date))[::100], date[::100], rotation=45)
            plt.title("%s vs %s" % (act, pred), fontsize="14")
            plt.ylim(0, 10500)

    ani = animation.FuncAnimation(fig, update, interval=100, frames=len(vis) + 1)
    ani.save(ani_name, writer="ffmpeg")


def preprocessing(X):
    from skynet.data_handling.preprocessing import PreProcessor

    preprocess = PreProcessor(norm=False, binary=False)
    preprocess.fit(X_test=X.iloc[:, :-1], y_test=X.iloc[:, -1])
    X = pd.concat([preprocess.X_test, preprocess.y_test], axis=1)

    return X


def main():
    import matplotlib.pyplot as plt
    from skynet import DATA_PATH
    from skynet.data_handling import read_learning_data
    from skynet.data_handling import split_time_series

    icao = "RJFT"
    date = "20180830"
    os.makedirs(os.getcwd() + "/confusion_matrix", exist_ok=True)
    os.makedirs(os.getcwd() + "/%s/%s" % (icao, date), exist_ok=True)

    X = read_learning_data(DATA_PATH + "/skynet/test_%s.pkl" % icao)

    # 前処理
    # X = preprocessing(X)

    # 時系列でデータを分割
    spX = split_time_series(X, level="month", period=2)

    # 期間別評価
    sppred = predict_by_period(spX, icao, smooth=False)

    # metar
    metar = set_visibility_metar(icao)
    metar = sync_values(base=metar, x=X[["date", "visibility_rank"]], key="date")
    spmetar = split_time_series(metar, level="month", period=2)

    # human edit
    he = set_visibility_human_edit(icao)
    sphe = split_time_series(he, level="month", period=2)

    # 期間別混同行列
    cfm_he, cfm_ml, cfm_heml = make_conf_mat_by_period(metar=spmetar, he=sphe, ml=sppred,
                                                       rank=3, nodal=[0, 800, 1000, 10000])

    cfm_he1y = 0
    cfm_ml1y = 0
    cfm_heml1y = 0
    for key in spmetar:
        cfm_he1y += cfm_he[key]
        cfm_ml1y += cfm_ml[key]
        cfm_heml1y += cfm_heml[key]

    cfm_he1y.to_html(os.getcwd() + "/confusion_matrix/metar_human_%s.html" % icao)
    cfm_ml1y.to_html(os.getcwd() + "/confusion_matrix/metar_ml_%s.html" % icao)
    cfm_heml1y.to_html(os.getcwd() + "/confusion_matrix/human_ml_%s.html" % icao)

    print(cfm_he1y)
    print()
    print(cfm_ml1y)
    print()
    print(cfm_heml1y)
    print()

    plt.show()


if __name__ == "__main__":
    main()
