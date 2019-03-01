def get_grib(layer, files, forecast_time, param, level=None):
    import pygrib
    grb = None
    if layer == 'surface':
        for f in files:
            grbs = pygrib.open(f)
            if check_grib(grbs, ft=int(forecast_time), param=param):
                grb = grbs.select(forecastTime=int(forecast_time), parameterName=param)[0]
            grbs.close()
    else:
        for f in files:
            grbs = pygrib.open(f)
            if check_grib(grbs, ft=int(forecast_time), param=param, level=int(level)):
                grb = grbs.select(forecastTime=int(forecast_time), parameterName=param, level=int(level))[0]
            grbs.close()

    return grb


def check_grib(grbs, ft, param, level=None):
    if level is None:
        try:
            grbs.select(forecastTime=ft, parameterName=param)
            return True
        except ValueError:
            return False
    else:
        try:
            grbs.select(forecastTime=ft, parameterName=param, level=level)
            return True
        except ValueError:
            return False


def msm_airport_ft0(icaos):
    import re
    import glob
    import gc
    import pickle
    import pygrib
    import skynet.nwp2d as npd
    from skynet import MSM_INFO, MSM_DATA_DIR

    latlon = npd.msm.get_airport_latlon(icaos)
    sf_latlon_idx = npd.msm.latlon_to_indices(latlon, layer='surface')
    up_latlon_idx = npd.msm.latlon_to_indices(latlon, layer='upper')

    tagid_list = [tagid for tagid in MSM_INFO.keys() if re.match(r'4002200', tagid)]
    tagid_list.sort()

    df_airports = {icao: npd.NWPFrame() for icao in icaos}
    for icao in icaos:
        for tagid in tagid_list:
            meta = MSM_INFO[tagid]

            layer = meta['layer']

            path = '%s/%s/bt%s/vt%s%s' % (
                MSM_DATA_DIR,
                layer,
                meta['base time'],
                meta['first validity time'],
                meta['last validity time']
            )

            path_list = glob.glob('%s/201*' % path)
            path_list.sort()

            for p in path_list:
                print(p)
                msm_files = glob.glob('%s/201*' % p)
                msm_files.sort()
                for f in msm_files:
                    grbs = pygrib.open(f)
                    if layer == 'surface':
                        grb = grbs.select()[0]
                        if grb is None:
                            continue
                        date = grb.validDate.strftime("%Y-%m-%d %H:%M")
                        param = grb.parameterName

                        lat = sf_latlon_idx[icao][0]
                        lon = sf_latlon_idx[icao][1]
                        df_airports[icao].loc[date, param] = grb.values[lat, lon]

                        del grb
                        gc.collect()

                    if layer == 'upper':
                        grb = grbs.select()[0]
                        if grb is None:
                            continue
                        date = grb.validDate.strftime("%Y-%m-%d %H:%M")
                        param = grb.parameterName[:4] + str(grb.level)

                        lat = up_latlon_idx[icao][0]
                        lon = up_latlon_idx[icao][1]
                        df_airports[icao].loc[date, param] = grb.values[lat, lon]

                        del grb
                        gc.collect()

                    grbs.close()

        df_airports[icao].to_csv('/Users/makino/PycharmProjects/SkyCC/data/msm_airport/%s.csv' % icao)
    pickle.dump(df_airports, open('/Users/makino/PycharmProjects/SkyCC/data/all_airport.pkl', 'wb'))


def msm_airport_xy(icao, metar_dir, msm_dir, save_dir):
    import re
    import pandas as pd
    import skynet.nwp2d as npd
    import skynet.datasets as skyds

    # metar読み込み
    with open('%s/head.txt' % metar_dir, 'r') as f:
        header = f.read()
    header = header.split(sep=',')

    data15 = pd.read_csv('%s/2015/%s.txt' % (metar_dir, icao), sep=',')
    data16 = pd.read_csv('%s/2016/%s.txt' % (metar_dir, icao), sep=',')
    data17 = pd.read_csv('%s/2017/%s.txt' % (metar_dir, icao), sep=',', names=header)

    metar_data = pd.concat([data15, data16, data17])
    metar_data = npd.NWPFrame(metar_data)

    metar_data.strtime_to_datetime('date', '%Y%m%d%H%M%S', inplace=True)
    metar_data.datetime_to_strtime('date', '%Y-%m-%d %H:%M', inplace=True)
    metar_data.drop_duplicates('date', inplace=True)
    metar_data.index = metar_data['date'].values

    metar_keys = ['date', 'visibility', 'str_cloud']
    metar_data = metar_data[metar_keys]
    metar_data['visibility_rank'] = skyds.to_visrank(metar_data['visibility'])

    # MSM読み込み
    msm_data = pd.read_csv('%s/%s.csv' % (msm_dir, icao))

    msm_data.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
    msm_data.index = msm_data['date'].values
    msm_data.sort_index(inplace=True)

    fets = skyds.get_init_features()
    target = skyds.get_init_target()
    X = npd.NWPFrame(pd.concat([msm_data[fets], metar_data[target]], axis=1))
    X.dropna(inplace=True)
    X.strtime_to_datetime('date', '%Y-%m-%d %H:%M', inplace=True)
    X.datetime_to_strtime('date', '%Y%m%d%H%M', inplace=True)
    X = X[fets + target]

    date = [d for d in X.index if not re.match('2017', d)]
    train = npd.NWPFrame(X.loc[date])
    train['date'] = train.index
    df_date = train.split_strcol(
        'date', ['year', 'month', 'day', 'hour', 'min'], pattern=r'[-\s:]'
    )[['year', 'month', 'day', 'hour', 'min']]
    train = pd.concat([df_date, train], axis=1)
    train.drop('date', axis=1, inplace=True)
    train.to_csv('%s/%s.csv' % (save_dir, icao), index=False)


def main():
    from skynet import DATA_DIR

    # jp_icaos = msm.get_jp_icaos()
    jp_icaos = [
        # 'RJOT',
        # 'RJAA',
        # 'RJBB',
        'RJCC',
        'RJCH',
        'RJFF',
        'RJFK',
        'RJGG',
        'RJNK',
        'RJOA',
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
    ]

    # msm_airport_ft0(jp_icaos)

    icao = 'RJBB'
    metar_dir = '%s/metar/airport' % DATA_DIR
    msm_dir = '%s/MSM/airport' % DATA_DIR
    save_dir = '%s/MSM/airport.process' % DATA_DIR

    msm_airport_xy(icao, metar_dir, msm_dir, save_dir)


if __name__ == '__main__':
    main()
