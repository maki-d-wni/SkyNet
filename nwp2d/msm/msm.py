import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pygrib
from skynet import AIRPORT_LATLON, MSM_BBOX, MSM_SHAPE

try:
    from mpl_toolkits.basemap import Basemap
except KeyError:
    import os
    import conda

    conda_file_dir = conda.__file__
    conda_dir = conda_file_dir.split('lib')[0]
    proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
    os.environ['PROJ_LIB'] = proj_lib

    from mpl_toolkits.basemap import Basemap

from skynet import MSM_BBOX, MSM_SHAPE


class DrawerBase(object):
    def __init__(self):
        self._figsize = None
        self._top = None
        self._bottom = None
        self._right = None
        self._left = None
        self._title_fontsize = None
        self._projection = 'cyl'
        self._resolution = 'l'
        self._coastlines_color = 'gray'
        self._countries_color = 'gray'
        self._cmap = 'jet'

    @property
    def figsize(self):
        return self._figsize

    @figsize.setter
    def figsize(self, figsize):
        self._figsize = figsize

    @property
    def top(self):
        return self._top

    @top.setter
    def top(self, top):
        self._top = top

    @property
    def bottom(self):
        return self._bottom

    @bottom.setter
    def bottom(self, bottom):
        self._bottom = bottom

    @property
    def right(self):
        return self._right

    @right.setter
    def right(self, right):
        self._right = right

    @property
    def left(self):
        return self._left

    @left.setter
    def left(self, left):
        self._left = left

    @property
    def title_fontsize(self):
        return self._title_fontsize

    @title_fontsize.setter
    def title_fontsize(self, title_fontsize):
        self._title_fontsize = title_fontsize

    @property
    def projection(self):
        return self._projection

    @projection.setter
    def projection(self, projection):
        self._projection = projection

    @property
    def resolution(self):
        return self._resolution

    @resolution.setter
    def resolution(self, resolution):
        self._resolution = resolution

    @property
    def coastlines_color(self):
        return self._coastlines_color

    @coastlines_color.setter
    def coastlines_color(self, coastlines_color):
        self._coastlines_color = coastlines_color

    @property
    def countries_color(self):
        return self._countries_color

    @countries_color.setter
    def countries_color(self, countries_color):
        self._countries_color = countries_color

    @property
    def cmap(self):
        return self._cmap

    @cmap.setter
    def cmap(self, cmap):
        self._cmap = cmap


class MSMDrawer(DrawerBase):
    def __init__(self, msm_files=None, params=None, levels=None, forecast_time=None):
        super().__init__()
        self.lat1 = MSM_BBOX[1]
        self.lat2 = MSM_BBOX[3]
        self.lon1 = MSM_BBOX[0]
        self.lon2 = MSM_BBOX[2]

        self.params = params
        self.levels = levels
        self.forecast_time = forecast_time

        if msm_files is not None:
            self.__grbs_list = read_grib(msm_files, params=params, levels=levels, forecast_time=forecast_time)

    def run(self):
        us = {}
        vs = {}
        for grb in self.__grbs_list:
            if grb['parameterName'] == 'Pressure reduced to MSL':
                self.pressure(grb['values'], grb['parameterName'], grb['level'], grb['forecastTime'])
            elif grb['parameterName'] == 'u-component of wind':
                us[grb['forecastTime']] = {'values': grb['values'], 'level': grb['level']}
            elif grb['parameterName'] == 'v-component of wind':
                vs[grb['forecastTime']] = {'values': grb['values'], 'level': grb['level']}
            elif grb['parameterName'] == 'Temperature':
                self.temperature(grb['values'], grb['parameterName'], grb['level'], grb['forecastTime'])
            elif grb['parameterName'] == 'Relative humidity':
                self.humidity(grb['values'], grb['parameterName'], grb['level'], grb['forecastTime'])
            elif grb['parameterName'] == 'Total precipitation':
                self.precipitation(grb['values'], grb['parameterName'], grb['level'], grb['forecastTime'])

        if len(us) and len(vs):
            for ft in self.forecast_time:
                u = us[ft]['values']
                v = vs[ft]['values']
                u_level = us[ft]['level']
                v_level = us[ft]['level']
                if u_level == v_level:
                    self.wind(u, v, 'Wind direction', u_level, ft)

    def pressure(self, pressure, param='Pressure reduced to MSL', level='surface', ft=0):
        fig = plt.figure(figsize=self.figsize)
        fig.add_subplot()
        fig.subplots_adjust(top=self.top, bottom=self.bottom, right=self.right, left=self.left)

        m = Basemap(
            projection="cyl",
            resolution="l",
            llcrnrlat=self.lat1,
            urcrnrlat=self.lat2,
            llcrnrlon=self.lon1,
            urcrnrlon=self.lon2
        )
        m.drawcoastlines(color='gray')
        # m.drawcountries(color='gray')

        interval = np.linspace(pressure.min(), pressure.max(), 16)
        lons, lats = self.__latlons(level)
        m.contour(lons, lats, pressure, levels=interval, latlon=True, cmap="jet")
        plt.xticks([120, 125, 130, 135, 140, 145, 150])
        plt.yticks([25, 30, 35, 40, 45])

        title = []
        grb_info = {'param': param, 'level': level, 'ft': ft}
        for key in grb_info:
            title.append('%s = %s' % (key, grb_info[key]))
        title = '\n'.join(title)

        plt.title(title, fontsize=self.title_fontsize)

        return fig

    def wind(self, u, v, param='Wind direction', level='surface', ft=0):
        fig = plt.figure(figsize=self.figsize)
        fig.add_subplot()
        fig.subplots_adjust(top=self.top, bottom=self.bottom, right=self.right, left=self.left)

        m = Basemap(
            projection="cyl",
            resolution="l",
            llcrnrlat=self.lat1,
            urcrnrlat=self.lat2,
            llcrnrlon=self.lon1,
            urcrnrlon=self.lon2
        )
        m.drawcoastlines(color='gray')
        # m.drawcountries(color='gray')

        # wind direction
        u_dir = u[::15, ::15].ravel()
        v_dir = v[::15, ::15].ravel()

        lons, lats = self.__latlons(level)
        x = lons[::15, ::15].ravel()
        y = lats[::15, ::15].ravel()

        plt.quiver(x, y, u_dir, v_dir, color="b")
        plt.xticks([120, 125, 130, 135, 140, 145, 150])
        plt.yticks([25, 30, 35, 40, 45])

        title = []
        grb_info = {'param': param, 'level': level, 'ft': ft}
        for key in grb_info:
            title.append('%s = %s' % (key, grb_info[key]))
        title = '\n'.join(title)

        plt.title(title, fontsize=self.title_fontsize)

        return fig

    def temperature(self, temperature, param='Temperature', level='surface', ft=0):
        fig = plt.figure(figsize=self.figsize)
        fig.add_subplot()
        fig.subplots_adjust(top=self.top, bottom=self.bottom, right=self.right, left=self.left)

        print(temperature.shape)

        m = Basemap(
            projection="cyl",
            resolution="l",
            llcrnrlat=self.lat1,
            urcrnrlat=self.lat2,
            llcrnrlon=self.lon1,
            urcrnrlon=self.lon2
        )
        m.drawcoastlines(color='gray')
        # m.drawcountries(color='gray')

        interval = np.arange(temperature.min(), temperature.max())
        mi = np.trunc(temperature.min()).astype(int)
        ma = np.ceil(temperature.max()).astype(int)
        delta = int((ma - mi) / 5)
        ticks = np.arange(mi, ma, delta)
        lons, lats = self.__latlons(level)
        plt.contourf(lons, lats, temperature, interval, cmap="jet", alpha=1.0)

        m.colorbar(ticks=ticks)

        title = []
        grb_info = {'param': param, 'level': level, 'ft': ft}
        for key in grb_info:
            title.append('%s = %s' % (key, grb_info[key]))
        title = '\n'.join(title)

        plt.title(title, fontsize=self.title_fontsize)

        return fig

    def humidity(self, humidity, param='Relative humidity', level='surface', ft=0):
        fig = plt.figure(figsize=self.figsize)
        fig.add_subplot()
        fig.subplots_adjust(top=self.top, bottom=self.bottom, right=self.right, left=self.left)

        m = Basemap(
            projection="cyl",
            resolution="l",
            llcrnrlat=self.lat1,
            urcrnrlat=self.lat2,
            llcrnrlon=self.lon1,
            urcrnrlon=self.lon2
        )
        m.drawcoastlines(color='gray')
        # m.drawcountries(color='gray')

        interval = np.arange(humidity.min(), humidity.max())
        ticks = np.arange(0, 100, 20)
        lons, lats = self.__latlons(level)
        plt.contourf(lons, lats, humidity, interval, cmap="jet", alpha=1.0)

        m.colorbar(ticks=ticks)

        title = []
        grb_info = {'param': param, 'level': level, 'ft': ft}
        for key in grb_info:
            title.append('%s = %s' % (key, grb_info[key]))
        title = '\n'.join(title)

        plt.title(title, fontsize=self.title_fontsize)

        return fig

    def precipitation(self, precipitation, param='Total precipitation', level='surface', ft=0):
        fig = plt.figure(figsize=self.figsize)
        fig.add_subplot()
        fig.subplots_adjust(top=self.top, bottom=self.bottom, right=self.right, left=self.left)

        m = Basemap(
            projection="cyl",
            resolution="l",
            llcrnrlat=self.lat1,
            urcrnrlat=self.lat2,
            llcrnrlon=self.lon1,
            urcrnrlon=self.lon2
        )
        m.drawcoastlines(color='gray')
        # m.drawcountries(color='gray')

        precipitation[precipitation < 1.] = 0

        interval = np.arange(1., precipitation.max())
        mi = np.trunc(precipitation.min()).astype(int)
        ma = np.ceil(precipitation.max()).astype(int)
        delta = int((ma - mi) / 5)
        ticks = np.arange(mi, ma, delta)
        lons, lats = self.__latlons(level)
        plt.contourf(lons, lats, precipitation, interval, cmap="jet", alpha=1.0)

        plt.xticks([120, 125, 130, 135, 140, 145, 150])
        plt.yticks([25, 30, 35, 40, 45])

        m.colorbar(ticks=ticks)

        title = []
        grb_info = {'param': param, 'level': level, 'ft': ft}
        for key in grb_info:
            title.append('%s = %s' % (key, grb_info[key]))
        title = '\n'.join(title)

        plt.title(title, fontsize=self.title_fontsize)

        return fig

    def __latlons(self, level):
        if level < 100:
            x0 = MSM_SHAPE['surface'][0]
            x1 = MSM_SHAPE['surface'][1]
            lats = np.linspace(self.lat2, self.lat1, x0)
            lons = np.linspace(self.lon1, self.lon2, x1)
            lons, lats = np.meshgrid(lons, lats)
        else:
            x0 = MSM_SHAPE['upper'][0]
            x1 = MSM_SHAPE['upper'][1]
            lats = np.linspace(self.lat2, self.lat1, x0)
            lons = np.linspace(self.lon1, self.lon2, x1)
            lons, lats = np.meshgrid(lons, lats)
        return lons, lats


def read_grib(files, params=None, levels=None, forecast_time=None):
    kwargs = {}
    if params is not None:
        kwargs['parameterName'] = params
    if levels is not None:
        kwargs['level'] = levels
    if forecast_time is not None:
        kwargs['forecastTime'] = forecast_time

    grbs_list = []
    for file in files:
        grbs = pygrib.open(file)

        try:
            selected_grbs = grbs.select(**kwargs)
        except ValueError:
            selected_grbs = None

        if selected_grbs is not None:
            for grb in selected_grbs:
                param = grb.parameterName
                level = grb.level
                vt = grb.forecastTime
                doc = {
                    'parameterName': param,
                    'level': level,
                    'forecastTime': vt,
                    'values': grb.values
                }
                grbs_list.append(doc)
        grbs.close()

    return grbs_list


def read_gribs_area(files, lon1, lat1, lon2, lat2):
    dict_grbs = {}
    for file in files:
        dict_grb = read_grib_area(file, lon1, lat1, lon2, lat2)

        for date_key in dict_grb.keys():
            if not (date_key in dict_grbs.keys()):
                dict_grbs = {date_key: {}}

            dict_grbs[date_key].update(dict_grb[date_key])

    return dict_grbs


def read_grib_airport(file, icao, layer):
    latlon = get_airport_latlon([icao])
    idx_latlon = airport_latlon_to_indices(latlon, layer)

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


def read_grib_area(file, lon1, lat1, lon2, lat2):
    grbs = pygrib.open(file)
    grbs = grbs.select()

    level = grbs[0].level
    if (level >= 0) and (level < 100):
        layer = 'surface'
    else:
        layer = 'upper'

    idx_lat1, idx_lon1 = latlon_to_index(lat1, lon1, layer)
    idx_lat2, idx_lon2 = latlon_to_index(lat2, lon2, layer)

    dict_grbs = {}
    for grb in grbs:
        ft = grb.forecastTime
        pn = grb.parameterName
        date = grb.validDate.strftime('%Y-%m-%d %H:%M')
        if layer == 'upper':
            pn += '_' + str(grb.level)

        if not (date in dict_grbs.keys()):
            dict_grbs = {date: {}}

        dict_grbs[date][pn] = grb.values[idx_lat2:idx_lat1, idx_lon1:idx_lon2]

    return dict_grbs


def get_jp_icaos():
    import re
    from skynet import ICAOS, AIRPORT_LATLON, MSM_BBOX

    icaos = [icao for icao in ICAOS if (re.match('RJ', icao) or re.match('RO', icao))]
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
    latlon = {
        icao: {
            'lat': AIRPORT_LATLON[icao]["lat"],
            'lon': AIRPORT_LATLON[icao]["lon"]
        } for icao in icaos if icao in AIRPORT_LATLON.keys()
    }

    return latlon


def latlon_to_index(lat, lon, layer):
    lat1_grb = MSM_BBOX[1]
    lat2_grb = MSM_BBOX[3]
    lon1_grb = MSM_BBOX[0]
    lon2_grb = MSM_BBOX[2]

    if layer == 'surface':
        n_lats = MSM_SHAPE['surface'][0]
        n_lons = MSM_SHAPE['surface'][1]
    else:
        n_lats = MSM_SHAPE['upper'][0]
        n_lons = MSM_SHAPE['upper'][1]

    idx_lat = round((n_lats - 1) * (1 - (lat - lat1_grb) / (lat2_grb - lat1_grb)))
    idx_lon = round((n_lons - 1) * (lon - lon1_grb) / (lon2_grb - lon1_grb))

    return idx_lat, idx_lon


def airport_latlon_to_indices(latlon, layer):
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
            round((n_lats - 1) * (1 - (AIRPORT_LATLON[icao]['lat'] - lat1_grb) / (lat2_grb - lat1_grb))),
            round((n_lons - 1) * (AIRPORT_LATLON[icao]['lon'] - lon1_grb) / (lon2_grb - lon1_grb))
        )
        for icao in icaos
    }

    return idx_latlon


def main():
    import glob
    import pygrib

    msm_dir = '/Users/makino/PycharmProjects/SkyCC/data/MSM/raw/surface/' \
              'bt00/vt0015/20150101_000000.000'
    msm_files = glob.glob('%s/*_*.*.*' % msm_dir)
    msm_files.sort()

    dict_grbs = read_gribs_area(msm_files, lon1=137.75, lat1=36.7, lon2=139.75, lat2=38.7)
    print(dict_grbs)

    '''
    grbs = pygrib.open(msm_files[0])
    for grb in grbs:
        latlon = grb.latlons()
        lat1, lat2, lon1, lon2 = latlon[0].min(), latlon[0].max(), latlon[1].min(), latlon[1].max()
        # print(lat1, lat2, lon1, lon2)
        lat1_ngt, lat2_ngt, lon1_ngt, lon2_ngt = \
            latlon[0][109, 142], latlon[0][89, 158], latlon[1][109, 142], latlon[1][89, 158]
        print(lat1_ngt, lat2_ngt, lon1_ngt, lon2_ngt)

    read_grib_airport(msm_files[0], 'RJAA', layer='surface')

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

    latlon = get_airport_latlon(jp_icaos)
    print(latlon)
    idx_latlon = airport_latlon_to_indices(latlon, layer='surface')
    print(idx_latlon)

    from skynet import MY_DIR
    print(MY_DIR)

    drawer = MSMDrawer(msm_files,
                       params=[
                           'Temperature'
                       ],
                       forecast_time=[0]
                       )
    drawer.run()
    plt.show()
    '''


if __name__ == "__main__":
    main()
