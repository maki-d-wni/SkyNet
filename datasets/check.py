def check_opengirb2():
    import glob
    import pygrib
    from skynet import MSM_INFO, MSM_DATA_DIR
    tagid_list = [tagid for tagid in MSM_INFO.keys() if re.match(r'4002200', tagid)]
    tagid_list.sort()

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

        path_list = glob.glob('%s/2017*' % path)
        path_list.sort()

        for p in path_list:
            msm_files = glob.glob('%s/201*' % p)
            for f in msm_files:
                grbs = pygrib.open(f)
                grbs.select()
                print(f)
                grbs.close()


def main():
    # jp_icaos = msm.get_jp_icaos()
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
        # 'RJBB',
        'RJCC',
        'RJCH',
        'RJFF',
        'RJFK',
        'RJGG',
        'RJNK',
        'RJOA',
    ]


if __name__ == '__main__':
    main()
