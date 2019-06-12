import os
import copy
import pickle
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

W = {
    "RJFK": [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 5]
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
    ],
    "RJAA": [
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 5]
    ],
    "RJBB": [
        [100, 20, 1, 1, 1, 1, 1, 1, 1],
        [100, 20, 1, 1, 1, 1, 1, 1, 1],
        [100, 20, 1, 1, 1, 1, 1, 1, 1],
        [100, 20, 1, 1, 1, 1, 1, 1, 1],
        [100, 20, 1, 1, 1, 1, 1, 1, 1],
        [100, 20, 1, 1, 1, 1, 1, 1, 1]
    ],
    "RJOT": [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1]
    ],
    "RJSK": [
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1]
    ],
    "RJSM": [
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1]
    ],
    "RJSN": [
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1]
    ],
    "RJSS": [
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1]
    ],
    "RJTT": [
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1]
    ],
    "RJOC": [
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1]
    ],
    "RJOO": [
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1]
    ],
    "RJCH": [
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1]
    ],
    "RJFF": [
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1]
    ],
    "RJGG": [
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1]
    ],
    "RJNK": [
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1]
    ],
    "RJOA": [
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1]
    ]
}


def confidence_factor(x, n_class):
    mv = np.zeros((len(x), n_class))
    idx = np.arange(len(x))
    for i in range(x.shape[1]):
        mv[idx, x[:, i].astype(int)] += 1

    confac = pd.DataFrame(mv)

    return confac


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
    import skynet.nwp2d as npd
    from skynet import DATA_DIR
    he = pd.read_csv(
        "%s/after/%s.csv" % (DATA_DIR, icao),
        names=["a", "b", "date", "c", "VIS_after", "d", "e", "f", "g", "h", "i"]
    )

    he = npd.NWPFrame(he[["date", "VIS_after"]])
    he.drop_duplicates("date", keep="first", inplace=True)

    vr = convert_visibility_rank(he["VIS_after"].values)
    he["visibility_rank"] = vr
    he.index = he.strtime_to_datetime('date', fmt='%Y%m%d%H%M')
    he.columns = ["date", "visibility", "visibility_rank"]

    return he


def set_visibility_area_forecast(icao):
    import pickle
    import skynet.nwp2d as npd
    from skynet import DATA_DIR

    af = pickle.load(open(DATA_DIR + "/skynet/metar.before.msm/test_%s.pkl" % icao, "rb"))
    af = af[["date", "VIS"]].rename(columns={"VIS": "visibility"})
    af["visibility_rank"] = convert_visibility_rank(af["visibility"].values)
    af = npd.NWPFrame(af)
    af.index = af.strtime_to_datetime('date', fmt='%Y%m%d%H%M')

    return af


def set_visibility_metar(icao):
    import skynet.nwp2d as npd
    from skynet import DATA_DIR
    metar = pd.read_csv(
        DATA_DIR + "/metar/airport_vis/metar_%s.csv" % icao
    )
    metar = npd.NWPFrame(metar)

    metar['visibility_rank'] = convert_visibility_rank(metar['visibility'].values)
    date = metar.strtime_to_datetime('date', fmt='%Y%m%d%H%M')
    metar.index = date

    return metar


def extract_different_index(s1, s2):
    s = pd.concat([s1, s2], axis=1).dropna()
    a = s.values[:, 0]
    b = s.values[:, 1]
    idx = np.where(a != b)[0]

    return idx


def sync_values(base, x):
    for k in x:
        base[k] = x[k]

    return base


def make_vis_table(metar=None, af=None, he=None, ml=None):
    def __append(df, values, kind):
        if "date" in values.keys():
            values = values.drop(["date"], axis=1)
        for key in values:
            if key in ["visibility", "visibility_rank"]:
                df["%s_%s" % (kind, key)] = values[key]
            else:
                df[key] = values[key]

        return df

    vis = pd.DataFrame()
    if metar is not None:
        vis = __append(vis, metar, kind="metar")
    if af is not None:
        vis = __append(vis, af, kind="AF")
    if he is not None:
        vis = __append(vis, he, kind="human")
    if ml is not None:
        vis = __append(vis, ml, kind="skynet")

    vis = vis.dropna()

    return vis


def convert_visibility_rank(vis):
    from skynet.datasets import base

    label = np.zeros(len(vis))
    v = list(base.get_init_vis_level().values()) + [100000]
    delta = np.diff(v)
    for i, d in enumerate(delta):
        indices = np.where((vis > v[i]) & (vis <= v[i] + d))[0]
        label[indices] = i
    return label


def adapt_visibility(v, v_min, v_max):
    return 9999 * (v - v_min) / (v_max - v_min)


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


def extend_conf_mat(y_true, y_pred, threats, true_name=None, pred_name=None):
    y1 = [str(t) for t in threats]
    y2 = [str(t) for t in threats if t != threats[0]] + [""]
    pair_threats = [[i, j] for i, j in zip(y1, y2)]

    rank = len(threats)
    delta = np.diff(np.array(threats + [1000000]).astype(int))

    t = np.zeros_like(y_true)
    p = np.zeros_like(y_pred)

    for i, d in enumerate(delta):
        t[(y_true >= threats[i]) & (y_true < (threats[i] + d))] = i
        p[(y_pred >= threats[i]) & (y_pred < (threats[i] + d))] = i

    cfm = np.zeros((rank, rank))
    for i in range(rank):
        for j in range(rank):
            cfm[i, j] = len(np.where((t == i) & (p == j))[0])

    if true_name is None:
        true_name = "Act"
    if pred_name is None:
        pred_name = "Pred"

    cfm = pd.DataFrame(
        cfm.astype(int),
        index=[
            [true_name for _ in range(rank)],
            [i.center(4, " ") + "-".center(3, " ") + j.center(4, " ") for i, j in pair_threats]
        ],
        columns=[
            [pred_name for _ in range(rank)],
            [i.center(4, " ") + "-".center(3, " ") + j.center(4, " ") for i, j in pair_threats]
        ]
    )

    return cfm


def make_conf_mat_by_period(metar, he, ml, threats):
    def __print_specified_term_result(string, pf1, prs, pps, length=12):
        space = [" " for _ in range(length - len(string))]
        for s in space:
            string += s
        print("%s: f1 = %0.3f, recall = %0.3f, precision = %0.3f" % (string, pf1, prs, pps))

    cfm_he = {}
    cfm_ml = {}
    cfm_heml = {}

    for key in metar:
        vis = make_vis_table(
            metar=metar[key],
            he=he[key],
            ml=ml[key]
        )

        cfm_he[key] = extend_conf_mat(y_true=vis["metar_visibility_rank"].values,
                                      y_pred=vis["human_visibility_rank"].values,
                                      threats=threats,
                                      true_name="metar",
                                      pred_name="human"
                                      )

        cfm_ml[key] = extend_conf_mat(y_true=vis["metar_visibility_rank"].values,
                                      y_pred=vis["skynet_visibility_rank"].values,
                                      threats=threats,
                                      true_name="metar",
                                      pred_name="skynet"
                                      )

        cfm_heml[key] = extend_conf_mat(y_true=vis["human_visibility_rank"].values,
                                        y_pred=vis["skynet_visibility_rank"].values,
                                        threats=threats,
                                        true_name="human",
                                        pred_name="skynet"
                                        )

        y = vis["metar_visibility_rank"].values
        p = vis["skynet_visibility_rank"].values
        f1, rs, ps = f1_recall_precision_score(y, p, threshold=1)
        __print_specified_term_result(key, f1, rs, ps)

    return cfm_he, cfm_ml, cfm_heml


def predict(X, clfs, w, smooth=False, confidence=False):
    n_clf = len(clfs)
    p_rf = np.zeros((len(X), n_clf))

    for i, clf in enumerate(clfs):
        p_rf[:, i] = clf.predict(X.values)

    # p = p_rf.mean(axis=1)
    p = majority_vote(p_rf, n_class=9, threshold=None)
    # p = deflected_mean(p_rf, w, threshold=6)

    if smooth:
        p = smoothing(p, ksize=6, threshold=5)

    if confidence:
        c = confidence_factor(p_rf, n_class=9)
        return p, c

    else:
        return p


def predict_by_period(X, clfs, icao, smooth=False, confidence=False):
    import skynet.datasets as skyds
    from sklearn.preprocessing import StandardScaler
    from skynet.nwp2d import NWPFrame

    pred = {}
    for i_term, key in enumerate(X):
        ss = StandardScaler()
        x = X[key]
        fets = skyds.get_init_features('long')

        x = x[fets]
        x = NWPFrame(ss.fit_transform(x), columns=x.keys())

        # モデルを用意

        if confidence:
            p, c = predict(x, clfs[key], W[icao][i_term], smooth, confidence)
            pred[key] = NWPFrame(copy.deepcopy(X[key][["date"]]))
            pred[key]["visibility"] = adapt_visibility(p, 0, 8)
            c["visibility_rank"] = p
            pred[key] = pd.concat([pred[key], c], axis=1)
            # pred[key].index = NWPFrame(pred[key].strtime_to_datetime('date', fmt='%Y-%m-%d %H:%M'))
        else:
            p = predict(x, clfs[key], W[icao][i_term], smooth, confidence)
            pred[key] = copy.deepcopy(X[key][["date"]])
            pred[key]["visibility"] = adapt_visibility(p, 0, 8)
            pred[key]["visibility_rank"] = p
            # pred[key].index = pred[key].strtime_to_datetime('date', fmt='%Y%m%d%H%M')

    return pred


def main():
    import skynet.nwp2d as npd
    import skynet.datasets as skyds
    from skynet import DATA_DIR, USER_DIR

    os.makedirs(os.getcwd() + "/confusion_matrix", exist_ok=True)

    icao = 'RJFK'
    '''
    'RJOT',
    'RJAA',
    'RJSC',
    'RJSI',
    'RJSK',
    'RJSM',
    'RJSN',
    'RJSS',
    'RJTT',
    'ROAH',
    'RJOC',
    'RJOO',
    # 'RJBB',
    'RJCC',
    'RJCH',
    'RJFF',
    'RJFK',
    'RJGG',
    'RJNK',
    'RJOA',
    '''

    data_dir = '%s/ARC-common/fit_input/JMA_MSM/vis' % DATA_DIR
    model_dir = '%s/ARC-common/fit_output/JMA_MSM/vis' % DATA_DIR
    model_name = 'GLOBAL_METAR-%s.vis' % icao
    data_name = 'GLOBAL_METAR-%s.vis' % icao
    month_keys = ['month:1-2', 'month:3-4', 'month:5-6', 'month:7-8', 'month:9-10', 'month:11-12']

    X = npd.NWPFrame(pd.read_csv('/Users/makino/PycharmProjects/SkyCC/data/skynet/test_%s.csv' % icao, sep=','))

    # 前処理
    # X = preprocessing(X)

    # print(msm_data)

    # 時系列でデータを分割
    spX = skyds.convert.split_time_series(X, X['month'], date_fmt='%m')

    # metar
    metar = set_visibility_metar(icao)
    # metar = sync_values(base=metar, x=X[["visibility_rank"]])
    spmetar = skyds.convert.split_time_series(
        metar,
        metar["date"],
        date_fmt='%Y%m%d%H%M'
    )

    # area_forecast
    af = set_visibility_area_forecast(icao)
    spaf = skyds.convert.split_time_series(
        af,
        date=af["date"],
        date_fmt='%Y%m%d%H%M'
    )

    # human edit
    he = set_visibility_human_edit(icao)
    sphe = skyds.convert.split_time_series(
        he,
        date=he["date"],
        date_fmt='%Y%m%d%H%M'
    )

    # モデルの準備
    '''
    clfs = {}
    model_dir = '%s/PycharmProjects/SkyCC/trained_models' % USER_DIR
    for i_term, key in enumerate(spX):
        clfs[key] = [
            pickle.load(
                open("%s/%s/forest/%s/rf%03d.pkl" % (model_dir, icao, key, i), "rb"))
            for i in range(N_CLF[i_term])
        ]

    clfs = {}
    for i_term, key in enumerate(spX):
        os.makedirs('%s/%s/stacking' % (model_dir, key), exist_ok=True)
        clfs[key] = pickle.load(
            open('%s/%s.pkl' % (model_dir, model_name), 'rb'))
    '''

    clfs = pickle.load(open('%s/%s.pkl' % (model_dir, model_name), 'rb'))

    # パラメーター
    confidence_list = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    confusion_matrix_threshold = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    score = pd.DataFrame()
    for t_num, threshold in enumerate(confidence_list):
        # 時系列毎の予測（コンフィデンスファクター付）
        sppred = predict_by_period(spX, clfs, icao, smooth=False, confidence=True)

        # Xのindexをdateに変換
        # X.index = X.strtime_to_datetime('date', fmt='%Y-%m-%d %H:%M')

        # 編集箇所チェック
        all_samples = 0
        for key in sphe:
            idx_edit = extract_different_index(sphe[key]["visibility_rank"],
                                               spaf[key]["visibility_rank"])
            edit = np.array(["" for _ in range(len(sphe[key]))])
            edit[idx_edit] = "*"
            sphe[key]["edit"] = edit
            all_samples += len(idx_edit)

        # 期間別vis_table
        spvis = {}
        drop_list = [
            "metar_visibility",
            "metar_visibility_rank",
            "human_visibility",
            "human_visibility_rank",
            "skynet_visibility",
            "skynet_visibility_rank",
            'tmp'
        ]
        for key in sppred:
            vis = make_vis_table(metar=spmetar[key], he=sphe[key], ml=sppred[key])
            vis["skynet"] = np.round(vis["skynet_visibility_rank"]).astype(int)
            vis["metar"] = vis["metar_visibility_rank"].astype(int)
            vis["human"] = vis["human_visibility_rank"].astype(int)

            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(vis['metar_visibility_rank'].values)
            plt.plot(vis['skynet_visibility_rank'].values)
            plt.show()

            vis = vis.rename(columns={"edit": "tmp"})
            vis["edit"] = vis["tmp"]
            vis = vis.drop(drop_list, axis=1)

            spvis[key] = vis
            spvis[key].insert(0, 'date', spX[key]['date'].values)

        # コンフィデンスファクターが閾値以下となる予測値を削除
        samples = 0
        for key in spvis:
            os.makedirs(os.getcwd() + "/confidence_factor/%s/%s" % (key, icao), exist_ok=True)
            confidence_map = spvis[key].loc[:, range(9)].values
            idx = confidence_map.argmax(axis=1)
            c_max = np.array([c[i] for i, c in zip(idx, confidence_map)])
            spvis[key] = spvis[key].iloc[c_max >= threshold]
            spvis[key].to_html(os.getcwd() + "/confidence_factor/%s/%s/%s_%s.html" % (key, icao, icao, threshold))

            edit = spvis[key]["edit"]
            samples += len([e for e in edit if e == "*"])

        print("all sample :", all_samples)
        print("samples :", samples)
        print("samples / all samples = %0.3f" % (samples / all_samples))
        print()

        # 期間別混同行列
        for key in sppred:
            idx = spvis[key].index
            sppred[key] = sppred[key].loc[idx]

        cfm_he, cfm_ml, cfm_heml = make_conf_mat_by_period(metar=spmetar, he=sphe, ml=sppred,
                                                           threats=confusion_matrix_threshold)

        cfm_he1y = 0
        cfm_ml1y = 0
        for key in cfm_he:
            cfm_he1y += cfm_he[key]
            cfm_ml1y += cfm_ml[key]

        os.makedirs(
            os.getcwd() + "/confusion_matrix/%dx%d/metar_vs_human/%s"
            % (len(confusion_matrix_threshold), len(confusion_matrix_threshold), icao), exist_ok=True
        )
        os.makedirs(
            os.getcwd() + "/confusion_matrix/%dx%d/metar_vs_ml/%s"
            % (len(confusion_matrix_threshold), len(confusion_matrix_threshold), icao), exist_ok=True
        )

        cfm_he1y.to_html(
            os.getcwd() + "/confusion_matrix/%dx%d/metar_vs_human/%s/%s_%s.html"
            % (len(confusion_matrix_threshold), len(confusion_matrix_threshold), icao, icao, threshold),
        )
        cfm_ml1y.to_html(
            os.getcwd() + "/confusion_matrix/%dx%d/metar_vs_ml/%s/%s_%s.html"
            % (len(confusion_matrix_threshold), len(confusion_matrix_threshold), icao, icao, threshold),
        )

        print(cfm_he1y)
        print()
        print(cfm_ml1y)
        print()

        mat = cfm_ml1y.values
        rs = mat[0, 0] / (mat[0, 0] + mat[0, 1])
        ps = mat[0, 0] / (mat[0, 0] + mat[1, 0])
        f1 = 2 * rs * ps / (rs + ps)
        print("total: f1 = %0.3f, recall = %0.3f, precision = %0.3f" % (f1, rs, ps))
        print()

        score = score.append([
            [
                threshold,
                # all_samples,
                # samples,
                # samples / all_samples,
                f1,
                rs,
                ps
            ]
        ])

    score.columns = ["confidence",
                     # "number of edit",
                     # "edit reduction", "%",
                     "f1",
                     "recall",
                     "precision"]
    score = score.round(3)
    # score["%"] *= 100
    print(score)

    os.makedirs(os.getcwd() + "/score", exist_ok=True)
    score.to_html(os.getcwd() + "/score/%s.html" % icao, index=False)


if __name__ == "__main__":
    main()
