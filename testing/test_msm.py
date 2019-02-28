def main():
    import glob
    import pygrib
    import matplotlib.pyplot as plt

    import skynet.nwp2d as npd

    layer = 'surface'

    msm_dir = '/Users/makino/PycharmProjects/SkyCC/data/grib2/MSM/%s/bt00/vt0015/20150101_000000.000' % layer
    msm_files = glob.glob('%s/2015*' % msm_dir)
    msm_files.sort()
    print('ファイル一覧\n各ファイルに一つのparameter, level, validity timeが入っている')
    print(msm_files)
    print()

    msm_files = ['/Users/makino/PycharmProjects/SkyCC/data/tss_sky_ml/400220115/20180704_030000.grib2']

    grbs = pygrib.open(msm_files[0])
    grb = grbs.select()[0]  # リストで返ってくる
    param = grb.parameterName
    level = grb.level
    vt = grb.forecastTime

    print('parameter -> %s' % param)
    print('level -> %d' % level)
    print('validity time -> %d' % vt)
    print()

    latlon = grb.latlons()
    print('latlon')
    print(latlon)

    # 値を取得
    val = grb.values
    print('values shape ->', val.shape)  # surfaceの場合shapeは(505, 481), upperの場合shapeは(253, 241)
    print()

    # ここからポイントデータを取得
    icaos = npd.msm.get_jp_icaos()  # 日本の空港コード取得
    latlon_icao = npd.msm.get_airport_latlon(icaos)  # 空港の緯度、経度取得
    idx_latlon = npd.msm.latlon_to_indices(latlon_icao, layer=layer)  # 取得した緯度、経度をindexに変換

    print('緯度、経度')
    print(latlon_icao)
    print('index')
    print(idx_latlon)
    print()

    print('nearest 確認')
    plt.imshow(val)
    for icao in icaos:
        idx_lat = idx_latlon[icao][0]
        idx_lon = idx_latlon[icao][1]
        print('%s,' % icao,
              '緯度 -> %f,' % latlon[0][idx_lat][idx_lon],
              '経度 -> %f,' % latlon[1][idx_lat][idx_lon],
              'value -> %f' % val[idx_lat][idx_lon]
              )

        plt.scatter(idx_latlon[icao][1], idx_latlon[icao][0], c='r', s=8)
        plt.text(idx_latlon[icao][1] + 5, idx_latlon[icao][0] - 5, '%s' % icao)
    plt.show()


if __name__ == '__main__':
    main()
