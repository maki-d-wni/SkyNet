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


def read_airport(file, icao, layer):
    import pygrib
    import pandas as pd
    latlon = get_airport_latlon([icao])
    idx_latlon = latlon_to_indices(latlon, layer)

    grbs = pygrib.open(file)

    df = pd.DataFrame()
    for grb in grbs:
        ft = grb.forecastTime
        date = grb.validDate.strftime("%Y-%m-%d %H:%M")
        df.loc[ft, 'date'] = date
        pn = grb.parameterName
        if layer == 'upper':
            pn = pn[:4] + str(grb.level)
        lat = idx_latlon[icao][0]
        lon = idx_latlon[icao][1]
        df.loc[ft, pn] = grb.values[lat, lon]

    return df


def concat_surface_upper(surface, upper):
    import pandas as pd
    icaos = list(surface.keys())
    df_grbs = {icao: None for icao in icaos}
    for icao in icaos:
        df_grbs[icao] = pd.concat([surface[icao], upper[icao]], axis=1)

    return df_grbs


def plot_forecast_map(file_path, layer, params, forecast_time, level=None, alpha=1., show=True, save_path=None):
    import pygrib
    from mpl_toolkits.basemap import Basemap
    grbs = pygrib.open(file_path)

    lat1 = 22.4
    lat2 = 47.6
    lon1 = 120
    lon2 = 150

    if hasattr(forecast_time, "__iter__"):
        fcst = forecast_time
    else:
        fcst = [forecast_time]

    for ft in fcst:
        fig = plt.figure(figsize=(12, 12))
        fig.add_subplot()
        fig.subplots_adjust(top=1, bottom=0., right=1.0, left=0.)

        m = Basemap(projection="cyl", resolution="l", llcrnrlat=lat1, urcrnrlat=lat2, llcrnrlon=lon1, urcrnrlon=lon2)
        # m.drawcoastlines(color='lightgray')
        # m.drawcountries(color='lightgray')
        # m.fillcontinents(color="white", lake_color="white")
        # m.drawmapboundary(fill_color="white")
        m.bluemarble()

        if type(params) == str:
            params = [params]

        for param in params:
            if layer == "surface":
                if param in ["Wind speed", "Wind direction"]:
                    grb = grbs.select(forecastTime=ft,
                                      parameterName=["u-component of wind", "v-component of wind"])
                else:
                    grb = grbs.select(forecastTime=ft, parameterName=param)
                level = "surface"
            else:
                if param in ["Wind speed", "Wind direction"]:
                    grb = grbs.select(forecastTime=ft,
                                      parameterName=["u-component of wind", "v-component of wind"],
                                      level=level)
                else:
                    grb = grbs.select(forecastTime=ft, parameterName=param, level=level)

            lats, lons = grb[0].latlons()

            if param == "Wind direction":
                u = grb[0].values[::10, ::10].ravel()
                v = grb[1].values[::10, ::10].ravel()

                x = lons[::10, ::10].ravel()
                y = lats[::10, ::10].ravel()

                plt.quiver(x, y, u, v, color="lightgray")

            elif param == "Wind speed":
                u = grb[0].values
                v = grb[1].values

                val = np.sqrt(u ** 2 + v ** 2)

                interval = np.arange(val.min(), val.max())
                mi = np.trunc(val.min()).astype(int)
                ma = np.ceil(val.max()).astype(int)
                delta = int((ma - mi) / 5)
                ticks = np.arange(mi, ma, delta)
                plt.contourf(lons, lats, val, interval, latlon=True, cmap="jet", alpha=alpha)

                if len(params) == 1:
                    m.colorbar(location="bottom", ticks=ticks)

            elif param == "Total precipitation":
                val = grb[0].values

                interval = np.arange(1, 160)
                ticks = range(0, 160, 20)
                plt.contourf(lons, lats, val, interval, latlon=True, cmap="jet", alpha=alpha)

                if len(params) == 1:
                    m.colorbar(location="bottom", ticks=ticks)

            elif param in ["Pressure reduced to MSL", "Pressure"]:
                val = grb[0].values

                m.contour(lons, lats, val, latlon=True, cmap="jet")

                if len(params) == 1:
                    m.colorbar(location="bottom")
            else:
                val = grb[0].values

                interval = np.arange(val.min(), val.max())
                mi = np.trunc(val.min()).astype(int)
                ma = np.ceil(val.max()).astype(int)
                delta = int((ma - mi) / 5)
                ticks = np.arange(mi, ma, delta)
                plt.contourf(lons, lats, val, interval, latlon=True, cmap="jet", alpha=alpha)

                if len(params) == 1:
                    m.colorbar(location="bottom", ticks=ticks)

        plt.title("%s : level = %s : FT = %d" % (",".join(params), level, ft), fontsize=16)

        if save_path is not None:
            plt.savefig("%s/FT%02d.png" % (save_path, ft))

    if show:
        plt.show()


def animate_forecast_map(save_file, date, time, layer, params, level,
                         alpha=1., show=False, cache=True):
    import os
    import glob
    os.makedirs(IMAGE_PATH + "/tmp", exist_ok=True)

    if not cache:
        files = glob.glob(IMAGE_PATH + "/tmp/FT*")
        for f in files:
            os.remove(f)
        if layer == "surface":
            if "Total precipitation" in params:
                fts = range(15), range(15, 33)
            else:
                fts = range(16), range(16, 34)
            tids = GRIB["surface"]["tag_id"]["FT_0-15"], GRIB["surface"]["tag_id"]["FT_16-33"]
        else:
            fts = range(0, 16, 3), range(18, 34, 3)
            tids = GRIB["upper"]["tag_id"]["FT_0-15"], GRIB["upper"]["tag_id"]["FT_16-33"]

        for tid, ft in zip(tids, fts):
            plot_forecast_map(
                file_path='%s/%s/%s_%s.grib2' % (DATA_PATH, tid, date, time),
                layer=layer, params=params, forecast_time=ft, level=level, alpha=alpha,
                show=False, save_path=IMAGE_PATH + "/tmp"
            )
            for i in range(len(ft)):
                plt.close()

    files = glob.glob(IMAGE_PATH + "/tmp/FT*")
    files.sort()

    fig = plt.figure(figsize=(6, 6))
    fig.add_subplot()
    fig.subplots_adjust(top=1., bottom=0., right=1., left=0.)
    imgs = []

    for f in files:
        img = plt.imread(f)

        plt.axis("off")
        imgs.append([plt.imshow(img)])

    ani = animation.ArtistAnimation(fig, imgs, interval=100)
    ani.save(save_file, writer="ffmpeg")

    if show:
        plt.show()


def get_jp_icaos():
    import re
    from skynet import ICAOS, AIRPORT_LATLON, MSM_BBOX

    icaos = [icao for icao in ICAOS if re.match('RJ', icao)]
    icaos.sort()

    lat1_grb = MSM_BBOX[1]
    lat2_grb = MSM_BBOX[3]
    lon1_grb = MSM_BBOX[0]
    lon2_grb = MSM_BBOX[2]

    icaos = [icao for icao in icaos
             if (lat1_grb <= AIRPORT_LATLON[icao]['lat']) and (AIRPORT_LATLON[icao]['lat'] <= lat2_grb)
             and (lon1_grb <= AIRPORT_LATLON[icao]['lon']) and (AIRPORT_LATLON[icao]['lon'] <= lon2_grb)]

    return icaos


def get_airport_latlon(icaos):
    from skynet import AIRPORT_LATLON
    latlon = {
        icao: {
            'lat': AIRPORT_LATLON[icao]["lat"],
            'lon': AIRPORT_LATLON[icao]["lon"]
        } for icao in icaos if icao in AIRPORT_LATLON.keys()
    }

    return latlon


def latlon_to_indices(latlon, layer):
    from skynet import AIRPORT_LATLON, MSM_BBOX, MSM_SHAPE

    icaos = list(latlon.keys())

    lat1_grb = MSM_BBOX[1]
    lat2_grb = MSM_BBOX[3]
    lon1_grb = MSM_BBOX[0]
    lon2_grb = MSM_BBOX[2]

    if layer == "surface":
        n_lats = MSM_SHAPE['surface'][0]
        n_lons = MSM_SHAPE['surface'][1]
    else:
        n_lats = MSM_SHAPE['upper'][0]
        n_lons = MSM_SHAPE['upper'][1]

    idx_latlon = {
        icao: (
            round(n_lats * (AIRPORT_LATLON[icao]['lat'] - lat1_grb) / (lat2_grb - lat1_grb)),
            round(n_lons * (AIRPORT_LATLON[icao]['lon'] - lon1_grb) / (lon2_grb - lon1_grb))
        )
        for icao in icaos
    }

    return idx_latlon


def main():
    from skynet import MSM_BBOX, AIRPORT_LATLON
    """
    date = "20180704"
    time = "030000"

    # icaos = __get_icao()
    # icaos = ["RJFT", "RJFK", "RJOT", "RJCC"]
    icaos = ["RJFK"]

    # ポイントデータ抽出
    df_sf_grbs = read_airports(layer="surface", icaos=icaos, date=date)
    df_ul_grbs = read_airports(layer="upper", icaos=icaos, date=date)

    df_grbs = concat_surface_upper(df_sf_grbs, df_ul_grbs)

    print(df_grbs["RJFK"])
    """

    # path = '/home/maki-d/NFS/floria/part1/MSM/surface/bt00/vt0015/20150101_000000.000/20150101_000000.000.1'
    # df = read_airport(path, 'RJAA', layer='surface')

    jp_icaos = [
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
        'RJBB',
        'RJCC',
        'RJCH',
        'RJFF',
        'RJFK',
        'RJGG',
        'RJNK',
        'RJOA',
    ]

    print(MSM_BBOX)
    latlon = get_airport_latlon(jp_icaos)
    print(latlon)
    idx_latlon = latlon_to_indices(latlon, layer='surface')
    print(idx_latlon)


if __name__ == "__main__":
    main()
