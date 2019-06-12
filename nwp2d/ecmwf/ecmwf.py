import pygrib
import matplotlib.pyplot as plt


def main():
    data_dir = '/Users/makino/PycharmProjects/SkyCC/data/ECMWF'
    data_name = (
        '20190528_174914.000',
        '20190528_175914.000',
        '20190528_175915.000'
    )

    grbs = pygrib.open('%s/%s' % (data_dir, data_name[1]))
    grbs = grbs.select()

    for grb in grbs:
        v = grb.values
        latlon = grb.latlons()
        lat1, lat2, lon1, lon2 = latlon[0].min(), latlon[0].max(), latlon[1].min(), latlon[1].max()
        print(lat1, lat2, lon1, lon2)
        plt.imshow(v)
        plt.show()


if __name__ == '__main__':
    main()
