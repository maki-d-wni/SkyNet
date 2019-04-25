import pandas as pd
import skynet.nwp2d as npd
import datetime


def main():
    today = datetime.datetime.today()

    year = today.year
    month = today.month
    day = today.day
    bt = 0
    icao = 'RJFT'
    pred_name = 'GLOBAL_METAR-%s.vis' % icao

    pred_dir = '/home/maki-d/PycharmProjects/SkyCC/data/ARC-pred/pred_output/JMA_MSM/' \
               '%04d/%02d/%02d/%02d/vis/' % (year, month, day, bt)

    metar_dir = '/home/maki-d/PycharmProjects/SkyCC/data/evaluate/metar'

    pred = npd.NWPFrame(pd.read_csv('%s/%s.csv' % (pred_dir, pred_name)))
    pred.merge_strcol(['HEAD:YEAR', 'MON', 'DAY', 'HOUR'], 'date', inplace=True)
    pred.strtime_to_datetime(date_key='date', fmt='%Y%m%d%H', inplace=True)
    pred.datetime_to_strtime(date_key='date', fmt='%Y-%m-%d %H:%M', inplace=True)
    pred.index = pred['date'].values
    pred = pred['SKYNET-VIS']

    metar = pd.read_csv('%s/%s.csv' % (metar_dir, icao))
    metar.index = metar['date'].values
    metar = metar['visibility']
    metar = metar.groupby(level=0)
    metar = metar.last()
    metar.rename('METAR', inplace=True)
    metar.replace('CAVOK', 9999, inplace=True)

    vis = pd.concat([metar, pred], axis=1, sort=False)
    vis.dropna(inplace=True)
    print(vis)


if __name__ == '__main__':
    main()
