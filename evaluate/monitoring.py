import os
import re
import pandas as pd
import skynet.nwp2d as npd
import skynet.datasets as skyds


def get_metar_all(data_dir, icao):
    years = os.listdir(data_dir)
    years = [int(y) for y in years if re.match(r'\d{4}', y)]
    years.sort()

    metar = None
    for y in years:
        months = os.listdir('%s/%04d' % (data_dir, y))
        months = [int(m) for m in months if re.match(r'\d{2}', m)]
        months.sort()

        for m in months:
            days = os.listdir('%s/%04d/%02d' % (data_dir, y, m))
            days = [int(d) for d in days if re.match(r'\d{2}', d)]
            days.sort()

            for d in days:
                bt = os.listdir('%s/%04d/%02d/%02d/' % (data_dir, y, m, d))
                bt = [int(b) for b in bt if re.match(r'\d{2}', b)]
                bt.sort()

                for b in bt:
                    if os.path.exists('%s/%04d/%02d/%02d/%02d/%s.csv' % (data_dir, y, m, d, b, icao)):
                        new_metar = pd.read_csv('%s/%04d/%02d/%02d/%02d/%s.csv' % (data_dir, y, m, d, b, icao))
                        if metar is None:
                            metar = new_metar
                        else:
                            pd.concat([metar, new_metar])

    metar.index = metar['date']
    metar.sort_index(inplace=True)
    metar.drop_duplicates('date', inplace=True)

    return metar


def get_arc_pred_all(data_dir, icao):
    years = os.listdir(data_dir)
    years = [int(y) for y in years if re.match(r'\d{4}', y)]
    years.sort()

    pred = None
    for y in years:
        months = os.listdir('%s/%04d' % (data_dir, y))
        months = [int(m) for m in months if re.match(r'\d{2}', m)]
        months.sort()

        for m in months:
            days = os.listdir('%s/%04d/%02d' % (data_dir, y, m))
            days = [int(d) for d in days if re.match(r'\d{2}', d)]
            days.sort()

            for d in days:
                bt = os.listdir('%s/%04d/%02d/%02d' % (data_dir, y, m, d))
                bt = [int(b) for b in bt if re.match(r'\d{2}', b)]
                bt.sort()

                for b in bt:
                    if os.path.exists(
                            '%s/%04d/%02d/%02d/%02d/vis/GLOBAL_METAR-%s.vis.csv' % (data_dir, y, m, d, b, icao)):
                        new_pred = pd.read_csv(
                            '%s/%04d/%02d/%02d/%02d/vis/GLOBAL_METAR-%s.vis.csv' % (data_dir, y, m, d, b, icao))
                        if pred is None:
                            pred = new_pred
                        else:
                            pred = pd.concat([pred, new_pred])

    return pred


def split_date(pattern, datestr):
    date = re.split(pattern, datestr)
    return date[0], date[1], date[2], date[3], date[4]


def eval_one_forecast(metar: pd.DataFrame, pred: pd.DataFrame, save_dir):
    icao = metar['ICAO'][0]
    metar.index = metar['date']
    metar.sort_index(inplace=True)
    metar.drop_duplicates('date', inplace=True)

    pred = npd.NWPFrame(pred)
    pred_date_cols = ['HEAD:YEAR', 'MON', 'DAY', 'HOUR']
    for key in pred_date_cols:
        if key == 'HEAD:YEAR':
            pred[key] = pred[key].astype(str).str.pad(4, fillchar='0')
        else:
            pred[key] = pred[key].astype(str).str.pad(2, fillchar='0')
    pred.merge_strcol(pred_date_cols, 'date', inplace=True)
    pred.strtime_to_datetime('date', '%Y%m%d%H', inplace=True)
    pred.datetime_to_strtime('date', '%Y-%m-%d %H:%M', inplace=True)
    pred.index = pred['date']
    pred.sort_index(inplace=True)
    pred.drop_duplicates('date', inplace=True)

    vis = pd.concat([metar, pred], axis=1)
    vis = vis[['visibility', 'SKYNET-VIS']]
    vis.dropna(inplace=True)
    os.makedirs('%s/time_series' % save_dir, exist_ok=True)
    vis.to_html('%s/time_series/%s.html' % (save_dir, icao))

    vis_level = skyds.get_init_vis_level()
    steps = list(vis_level.values())

    cfm = conf_mat(vis['visibility'], vis['SKYNET-VIS'], steps)
    os.makedirs('%s/confusion_matrix' % save_dir, exist_ok=True)
    cfm.to_html('%s/confusion_matrix/%s.html' % (save_dir, icao))


def conf_mat(y_true, y_pred, steps):
    import numpy as np

    rank = len(steps)
    delta = np.diff(np.array(steps + [1000000]).astype(int))

    t = np.zeros_like(y_true)
    p = np.zeros_like(y_pred)

    for i, d in enumerate(delta):
        t[(y_true >= steps[i]) & (y_true < (steps[i] + d))] = i
        p[(y_pred >= steps[i]) & (y_pred < (steps[i] + d))] = i

    cfm = np.zeros((rank, rank))
    for i in range(rank):
        for j in range(rank):
            cfm[i, j] = len(np.where((t == i) & (p == j))[0])

    cfm = pd.DataFrame(
        cfm.astype(int),
        index=['Actual %d' % v for v in steps],
        columns=['Predicted %d' % v for v in steps]
    )

    return cfm


def calc_f1(y_true, y_pred):
    import numpy as np
    from sklearn.metrics import f1_score

    y_true_bn = np.zeros_like(y_true)
    y_true_bn[y_true < 1] = 1

    y_pred_bn = np.zeros_like(y_pred)
    y_pred_bn[y_pred < 1] = 1

    f1 = f1_score(y_true_bn, y_pred_bn)

    return f1


def main():
    from skynet import DATA_DIR

    icaos = ['RJCC', 'RJFK', 'RJFT', 'RJOT']

    for icao in icaos:
        print(icao)
        metar_dir = '%s/evaluate/metar' % DATA_DIR
        metar_name = '%s.csv' % icao
        arc_pred_dir = '%s/ARC-pred/pred_output/JMA_MSM' % DATA_DIR
        arc_pred_name = 'GLOBAL_METAR-%s.vis.csv' % icao

        metar_date = '2019-06-10 00:00'
        y, mo, d, h, mi = split_date('[-: ]', metar_date)
        metar_file = '%s/%s/%s/%s/%s/%s' % (metar_dir, y, mo, d, h, metar_name)
        metar = pd.read_csv(metar_file)

        pred_date = '2019-06-09 00:00'
        y, mo, d, h, mi = split_date('[-: ]', pred_date)
        pred_file = '%s/%s/%s/%s/%s/vis/%s' % (arc_pred_dir, y, mo, d, h, arc_pred_name)
        pred = pd.read_csv(pred_file)

        save_date = '2019-06-10 00:00'
        y, mo, d, h, mi = split_date('[-: ]', save_date)
        save_dir = '%s/evaluate/live/%s/%s/%s/%s' % (DATA_DIR, y, mo, d, h)
        os.makedirs(save_dir, exist_ok=True)
        eval_one_forecast(metar, pred, save_dir)


if __name__ == '__main__':
    main()
