def get_grib(layer, files, forecast_time, param, level=None):
    import pygrib
    grb = None
    if layer == 'surface':
        for f in files:
            grbs = pygrib.open(f)
            if check_grib(grbs, ft=int(forecast_time), param=param):
                grb = grbs.select(forecastTime=int(forecast_time), parameterName=param)[0]
    else:
        for f in files:
            grbs = pygrib.open(f)
            if check_grib(grbs, ft=int(forecast_time), param=param, level=int(level)):
                grb = grbs.select(forecastTime=int(forecast_time), parameterName=param, level=int(level))[0]

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


def main():
    import glob
    import pickle
    from skynet import MSM_INFO, MSM_DATA_DIR
    from skynet.nwp2d import NWPFrame
    from skynet.nwp2d import msm

    # jp_icaos = msm.get_jp_icaos()
    jp_icaos = [
        'RJAA',
        'RJBB',
        'RJCC',
        'RJCH',
        'RJFF',
        'RJFK',
        'RJGG',
        'RJNK',
        'RJOA',
        'RJOC',
        'RJOO',
        'RJOT',
        'RJSC',
        'RJSI',
        'RJSK',
        'RJSM',
        'RJSN',
        'RJSS',
        'RJTT',
        'ROAH'
    ]

    latlon = msm.get_airport_latlon(jp_icaos)
    sf_latlon_idx = msm.latlon_to_indices(latlon, layer='surface')
    up_latlon_idx = msm.latlon_to_indices(latlon, layer='upper')

    tagid_list = list(MSM_INFO.keys())

    df_airports = {icao: NWPFrame() for icao in jp_icaos}
    for icao in jp_icaos:
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
            path_list = glob.glob('%s/*' % path)
            path_list.sort()

            for p in path_list:
                print(p)
                msm_files = glob.glob('%s/*' % p)
                if layer == 'surface':
                    for param in MSM_INFO['parameter'][layer]:
                        grb = get_grib(layer, msm_files, forecast_time=0, param=param)
                        if grb is None:
                            continue
                        date = grb.validDate.strftime("%Y-%m-%d %H:%M")

                        lat = sf_latlon_idx[icao][0]
                        lon = sf_latlon_idx[icao][1]
                        df_airports[icao].loc[date, param] = grb.values[lat, lon]

                if layer == 'upper':
                    for param in MSM_INFO['parameter'][layer]:
                        for level in MSM_INFO['level'][layer]:
                            grb = get_grib(layer, msm_files, forecast_time=0, param=param, level=level)
                            if grb is None:
                                continue
                            date = grb.validDate.strftime("%Y-%m-%d %H:%M")
                            new_param = param[:4] + level

                            lat = up_latlon_idx[icao][0]
                            lon = up_latlon_idx[icao][1]
                            df_airports[icao].loc[date, new_param] = grb.values[lat, lon]

        df_airports[icao].to_csv('/home/maki-d/PycharmProjects/SkyCC/data/msm_airport/%s.csv' % icao)
    pickle.dump(df_airports, open('/home/maki-d/PycharmProjects/SkyCC/data/all_airport.pkl', 'wb'))


if __name__ == '__main__':
    main()
