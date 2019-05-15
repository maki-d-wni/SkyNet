import os
import re
import datetime
import numpy as np
import pandas as pd
import skynet.nwp2d as npd
from skynet import DATA_DIR


def calc_edit_rate(edit_ary):
    if len(edit_ary):
        return len(edit_ary[edit_ary == '*']) / len(edit_ary)
    else:
        return 0


def edit_rate_00_05():
    before_dir = '%s/before' % DATA_DIR
    after_dir = '%s/after' % DATA_DIR

    save_dir = '%s/evaluate/edit_rate' % DATA_DIR
    os.makedirs(save_dir, exist_ok=True)

    before_airports = os.listdir(before_dir)
    before_airports = {icao[:4] for icao in before_airports if re.match(r'^[A-Z]', icao)}

    after_airports = os.listdir(after_dir)
    after_airports = {icao[:4] for icao in after_airports if re.match(r'^[A-Z]', icao)}

    airports_list = list(before_airports & after_airports)
    airports_list.sort()

    # airports_series = pd.Series(airports_list, name='ICAO')
    # airports_series.to_csv('airport_list.csv', index=False)

    df_edit_all = pd.DataFrame()
    for icao in airports_list:
        print(icao)
        df_before = npd.NWPFrame(pd.read_csv('%s/%s.txt' % (before_dir, icao), sep=','))
        df_before.strtime_to_datetime(date_key='date', fmt='%Y%m%d%H%M', inplace=True)
        df_before.index = df_before['date'].values
        before_bt = [bt for bt in df_before['date'] if bt.hour == 6]
        vt_list = []
        for bt in before_bt:
            vt = [bt + datetime.timedelta(hours=t) for t in range(6)]
            vt_list += vt
        df_before_00_05 = df_before.loc[vt_list]
        df_before_00_05 = npd.NWPFrame(df_before_00_05)
        df_before_00_05.dropna(inplace=True)
        df_before_00_05.datetime_to_strtime(date_key='date', fmt='%Y-%m-%d %H:%M', inplace=True)
        df_before_00_05.index = df_before_00_05['date'].values

        h_after = [
            'ICAO', 'BASE', 'VALID', 'precipitation', 'visibility', 'ceiling', 'temperature', 'wind speed',
            'wind direction', 'WX_after', 'u4'
        ]
        df_after = npd.NWPFrame(pd.read_csv('%s/%s.csv' % (after_dir, icao), names=h_after))
        df_after.strtime_to_datetime(date_key='VALID', fmt='%Y%m%d%H%M', inplace=True)
        df_after.index = df_after['VALID'].values

        df_after_00_05 = npd.NWPFrame(
            df_after[['BASE', 'VALID', 'visibility', 'ceiling', 'wind speed', 'wind direction', 'WX_after']]
        )
        df_after_00_05.strtime_to_datetime(date_key='BASE', fmt='%Y%m%d%H%M', inplace=True)
        after_bt = list(df_after_00_05.drop_duplicates('BASE')['BASE'].values)
        vt_list = []
        for bt in after_bt:
            vt = [bt + np.timedelta64(t, 'h') for t in range(6)]
            vt_list += vt
        if df_after_00_05['BASE'][0] + np.timedelta64(6) < df_after_00_05['VALID'][0]:
            df_edit = pd.DataFrame([[
                icao,
                len(df_after_00_05),
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0
            ]], columns=['ICAO', 'All', 'Vis edit', 'VIS edit rate', 'CIG edit', 'CIG edit rate',
                         'WNDSPD edit', 'WNDSPD edit rate', 'WDIR edit', 'WDIR edit rate',
                         'WX edit', 'WX edit rate'])
            df_edit.to_csv('%s/edit_rate_%s.csv' % (save_dir, icao), index=False)
            df_edit_all = pd.concat([df_edit_all, df_edit])
            continue

        df_after_00_05 = df_after_00_05.loc[vt_list]
        df_after_00_05 = npd.NWPFrame(df_after_00_05)
        df_after_00_05.dropna(inplace=True)
        df_after_00_05.datetime_to_strtime(date_key='VALID', fmt='%Y-%m-%d %H:%M', inplace=True)
        df_after_00_05.drop_duplicates('VALID', inplace=True)
        df_after_00_05.index = df_after_00_05['VALID'].values

        # print(df_after[['BASE', 'visibility', 'ceiling', 'wind speed', 'wind direction', 'WX_after']])
        vis = pd.concat(
            [df_before_00_05, df_after_00_05],
            axis=1
        )
        vis = vis[
            [
                'ICAO', 'date', 'VIS', 'visibility', 'CLING', 'ceiling', 'WNDSPD', 'wind speed', 'WNDDIR',
                'wind direction', 'WX_after'
            ]
        ]

        vis.rename(columns={
            'VIS': 'VIS_before',
            'visibility': 'VIS_after',
            'CLING': 'CIG_before',
            'ceiling': 'CIG_after',
            'WNDSPD': 'WNDSPD_before',
            'wind speed': 'WNDSPD_after',
            'WNDDIR': 'WNDDIR_before',
            'wind direction': 'WNDDIR_after'
        }, inplace=True
        )
        vis.reset_index(drop=True, inplace=True)

        vis_range = [
            0,
            25,
            75,
            125,
            175,
            225,
            275,
            325,
            375,
            450,
            550,
            625,
            675,
            725,
            775,
            850,
            950,
            1050,
            1150,
            1250,
            1350,
            1450,
            1550,
            1650,
            1750,
            1900,
            2200,
            2700,
            3100,
            3600,
            4400,
            4900,
            5500,
            6500,
            7500,
            8500,
            9500,
            10000
        ]

        vis_values = [
            0,
            50,
            100,
            150,
            200,
            250,
            300,
            350,
            400,
            500,
            600,
            650,
            700,
            750,
            800,
            900,
            1000,
            1100,
            1200,
            1300,
            1400,
            1500,
            1600,
            1700,
            1800,
            2000,
            2400,
            3000,
            3200,
            4000,
            4800,
            5000,
            6000,
            7000,
            8000,
            9000,
            9999
        ]

        i = 0
        while True:
            idx = np.where((vis['VIS_before'] > vis_range[i]) & (vis['VIS_before'] <= vis_range[i + 1]))[0]
            vis.loc[idx, 'VIS_before'] = vis_values[i]
            i += 1
            if i == len(vis_values):
                break

        vis[['CIG_before', 'CIG_after']] *= 1 / 0.3048
        cig_range = [
            0,
            15,
            40,
            75,
            125,
            175,
            225,
            275,
            325,
            375,
            450,
            550,
            650,
            750,
            850,
            950,
            1050,
            1150,
            1250,
            1350,
            1450,
            1550,
            1650,
            1750,
            1850,
            1950,
            2050,
            2150,
            2250,
            2350,
            2450,
            2550,
            2650,
            2750,
            2850,
            2950,
            3250,
            3750,
            4500,
            100000
        ]

        cig_values = [
            0,
            30,
            50,
            100,
            150,
            200,
            250,
            300,
            350,
            400,
            500,
            600,
            700,
            800,
            900,
            1000,
            1100,
            1200,
            1300,
            1400,
            1500,
            1600,
            1700,
            1800,
            1900,
            2000,
            2100,
            2200,
            2300,
            2400,
            2500,
            2600,
            2700,
            2800,
            2900,
            3000,
            3500,
            4000,
            ''
        ]

        i = 0
        while True:
            idx = np.where((vis['CIG_before'] > cig_range[i]) & (vis['CIG_before'] <= cig_range[i + 1]))[0]
            vis.loc[idx, 'CIG_before'] = cig_values[i]
            i += 1
            if i == len(cig_values):
                break

        i = 0
        while True:
            idx = np.where((vis['CIG_after'] > cig_range[i]) & (vis['CIG_after'] <= cig_range[i + 1]))[0]
            vis.loc[idx, 'CIG_after'] = cig_values[i]
            i += 1
            if i == len(cig_values):
                break

        vis['WNDSPD_before'] *= 1 / 0.514444
        vis = vis.round({'WNDSPD_before': 0})

        vis = vis.round({'WNDDIR_before': -1})

        # vis.round('VIS_before', )

        '''
        VISはロジックあり
        CIGは小数点第３位で比較
        beforeの風速をノットに変更
        WNDDIRは1の位を四捨五入
        '''

        wx_list = [
            '',
            '',
            'RA',
            'SNRA',
            'SN',
            'SNRA',
            '',
            '',
            ''
        ]
        wx_telop = []
        for idx in df_before_00_05.index:
            wx_prob = df_before_00_05.loc[idx, [
                'WX_telop_100',
                'WX_telop_200',
                'WX_telop_300',
                'WX_telop_340',
                'WX_telop_400',
                'WX_telop_430',
                'WX_telop_500',
                'WX_telop_600',
                'WX_telop_610'
            ]]

            prc = df_before_00_05.loc[idx, 'PRCRIN_1HOUR_TOTAL']
            if (prc >= 5.) and (prc < 10.):
                wx = ''
            elif prc >= 10.:
                wx = '+'
            else:
                wx = '-'
            tnd = df_before_00_05.loc[idx, 'TNDSTM_prob']
            if tnd >= 50.:
                wx += 'TS'
            wx += wx_list[int(np.argmax(wx_prob.values))]

            v = df_before_00_05.loc[idx, 'VIS']
            tmpr = df_before_00_05.loc[idx, 'AIRTMP']
            if v < 1000.:
                wx += ' FG'
            elif (v < 1000.) and (tmpr < 0.):
                wx += ' FZFG'
            elif (v >= 1000.) and (v <= 5000.):
                wx += ' BR'

            wx_telop.append(wx)

        vis.insert(vis.shape[1] - 1, 'WX_before', wx_telop)
        vis['WX_after'] = vis['WX_after'].str.replace(' ', '')
        vis['WX_after'] = vis['WX_after'].str.replace('_', '-')
        vis.dropna(inplace=True)

        vis_edit = np.where(vis['VIS_before'] != vis['VIS_after'], '*', '')
        cig_edit = np.where(vis['CIG_before'] != vis['CIG_after'], '*', '')
        wspd_edit = np.where(vis['WNDSPD_before'] != vis['WNDSPD_after'], '*', '')
        wdir_edit = np.where(vis['WNDDIR_before'] != vis['WNDDIR_after'], '*', '')
        wx_edit = np.where(vis['WX_before'] != vis['WX_after'], '*', '')

        vis_edit_rate = calc_edit_rate(vis_edit)
        cig_edit_rate = calc_edit_rate(cig_edit)
        wspd_edit_rate = calc_edit_rate(wspd_edit)
        wdir_edit_rate = calc_edit_rate(wdir_edit)
        wx_edit_rate = calc_edit_rate(wx_edit)

        vis['VIS edit rate'] = vis_edit
        vis['CIG edit rate'] = cig_edit
        vis['WNDSPD edit rate'] = wspd_edit
        vis['WDIR edit rate'] = wdir_edit
        vis['WX edit rate'] = wx_edit

        # 時系列データが欲しい場合は書き足す
        columns = [
            'ICAO', 'date', 'VIS', 'visibility', 'CLING', 'ceiling', 'WNDSPD', 'wind speed', 'WNDDIR',
            'wind direction', 'WX_after'
        ]

        df_edit = pd.DataFrame([[
            icao,
            len(vis_edit),
            len(vis_edit[vis_edit == '*']),
            vis_edit_rate,
            len(cig_edit[cig_edit == '*']),
            cig_edit_rate,
            len(wspd_edit[wspd_edit == '*']),
            wspd_edit_rate,
            len(wdir_edit[wdir_edit == '*']),
            wdir_edit_rate,
            len(wx_edit[wx_edit == '*']),
            wx_edit_rate
        ]], columns=['ICAO', 'All', 'Vis edit', 'VIS edit rate', 'CIG edit', 'CIG edit rate',
                     'WNDSPD edit', 'WNDSPD edit rate', 'WDIR edit', 'WDIR edit rate',
                     'WX edit', 'WX edit rate'])

        df_edit.to_csv('%s/edit_rate_%s.csv' % (save_dir, icao), index=False)
        df_edit_all = pd.concat([df_edit_all, df_edit])

    df_edit_all = df_edit_all.round(3)
    df_edit_all.to_csv('%s/edit_rate_all_00_05.csv' % save_dir, index=False)


def edit_rate_06_23():
    before_dir = '%s/before' % DATA_DIR
    after_dir = '%s/after' % DATA_DIR

    save_dir = '%s/evaluate/edit_rate' % DATA_DIR
    os.makedirs(save_dir, exist_ok=True)

    before_airports = os.listdir(before_dir)
    before_airports = {icao[:4] for icao in before_airports if re.match(r'^[A-Z]', icao)}

    after_airports = os.listdir(after_dir)
    after_airports = {icao[:4] for icao in after_airports if re.match(r'^[A-Z]', icao)}

    airports_list = list(before_airports & after_airports)
    airports_list.sort()

    # airports_series = pd.Series(airports_list, name='ICAO')
    # airports_series.to_csv('airport_list.csv', index=False)

    df_edit_all = pd.DataFrame()
    for icao in airports_list:
        print(icao)
        df_before = npd.NWPFrame(pd.read_csv('%s/%s.txt' % (before_dir, icao), sep=','))
        df_before.strtime_to_datetime(date_key='date', fmt='%Y%m%d%H%M', inplace=True)
        df_before.index = df_before['date'].values
        before_bt = [bt for bt in df_before['date'] if bt.hour == 12]
        vt_list = []
        for bt in before_bt:
            vt = [bt + datetime.timedelta(hours=t) for t in range(18)]
            vt_list += vt

        df_before_06_23 = df_before.loc[vt_list]
        df_before_06_23 = npd.NWPFrame(df_before_06_23)
        df_before_06_23.dropna(inplace=True)
        df_before_06_23.datetime_to_strtime(date_key='date', fmt='%Y-%m-%d %H:%M', inplace=True)
        df_before_06_23.index = df_before_06_23['date'].values

        h_after = [
            'ICAO', 'BASE', 'VALID', 'precipitation', 'visibility', 'ceiling', 'temperature', 'wind speed',
            'wind direction', 'WX_after', 'u4'
        ]
        df_after = npd.NWPFrame(pd.read_csv('%s/%s.csv' % (after_dir, icao), names=h_after))
        df_after.strtime_to_datetime(date_key='VALID', fmt='%Y%m%d%H%M', inplace=True)
        df_after.index = df_after['VALID'].values

        df_after_06_23 = npd.NWPFrame(
            df_after[['BASE', 'VALID', 'visibility', 'ceiling', 'wind speed', 'wind direction', 'WX_after']]
        )
        df_after_06_23.strtime_to_datetime(date_key='BASE', fmt='%Y%m%d%H%M', inplace=True)
        after_bt = list(df_after_06_23.drop_duplicates('BASE')['BASE'].values)
        vt_list = []
        for bt in after_bt:
            vt = [bt + np.timedelta64(t, 'h') for t in range(6, 24)]
            vt_list += vt

        idx_check = True
        for v in vt_list:
            if v in df_after_06_23.index:
                idx_check = False
                break
        if idx_check:
            df_edit = pd.DataFrame([[
                icao,
                len(df_after_06_23),
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0
            ]], columns=['ICAO', 'All', 'Vis edit', 'VIS edit rate', 'CIG edit', 'CIG edit rate',
                         'WNDSPD edit', 'WNDSPD edit rate', 'WDIR edit', 'WDIR edit rate',
                         'WX edit', 'WX edit rate'])
            df_edit.to_csv('%s/edit_rate_%s.csv' % (save_dir, icao), index=False)
            df_edit_all = pd.concat([df_edit_all, df_edit])
            continue

        df_after_06_23 = df_after_06_23.loc[vt_list]
        df_after_06_23 = npd.NWPFrame(df_after_06_23)
        df_after_06_23.dropna(inplace=True)
        df_after_06_23.datetime_to_strtime(date_key='VALID', fmt='%Y-%m-%d %H:%M', inplace=True)
        df_after_06_23.drop_duplicates('VALID', inplace=True)
        df_after_06_23.index = df_after_06_23['VALID'].values

        # print(df_after[['BASE', 'visibility', 'ceiling', 'wind speed', 'wind direction', 'WX_after']])
        vis = pd.concat(
            [df_before_06_23, df_after_06_23],
            axis=1
        )
        vis = vis[
            [
                'ICAO', 'date', 'VIS', 'visibility', 'CLING', 'ceiling', 'WNDSPD', 'wind speed', 'WNDDIR',
                'wind direction', 'WX_after'
            ]
        ]

        '''
        if len(df_before_06_23) > len(df_after_06_23):
            vis_index = df_after_06_23.index
        else:
            vis_index = df_before_06_23.index
        vis = vis.loc[vis_index]
        '''

        vis.rename(columns={
            'VIS': 'VIS_before',
            'visibility': 'VIS_after',
            'CLING': 'CIG_before',
            'ceiling': 'CIG_after',
            'WNDSPD': 'WNDSPD_before',
            'wind speed': 'WNDSPD_after',
            'WNDDIR': 'WNDDIR_before',
            'wind direction': 'WNDDIR_after'
        }, inplace=True
        )

        # vis.reset_index(drop=True, inplace=True)

        vis_range = [
            0,
            25,
            75,
            125,
            175,
            225,
            275,
            325,
            375,
            450,
            550,
            625,
            675,
            725,
            775,
            850,
            950,
            1050,
            1150,
            1250,
            1350,
            1450,
            1550,
            1650,
            1750,
            1900,
            2200,
            2700,
            3100,
            3600,
            4400,
            4900,
            5500,
            6500,
            7500,
            8500,
            9500,
            10000
        ]

        vis_values = [
            0,
            50,
            100,
            150,
            200,
            250,
            300,
            350,
            400,
            500,
            600,
            650,
            700,
            750,
            800,
            900,
            1000,
            1100,
            1200,
            1300,
            1400,
            1500,
            1600,
            1700,
            1800,
            2000,
            2400,
            3000,
            3200,
            4000,
            4800,
            5000,
            6000,
            7000,
            8000,
            9000,
            9999
        ]

        i = 0
        while True:
            idx = np.where((vis['VIS_before'] > vis_range[i]) & (vis['VIS_before'] <= vis_range[i + 1]))[0]
            idx = vis.index[idx]
            vis.loc[idx, 'VIS_before'] = vis_values[i]
            i += 1
            if i == len(vis_values):
                break

        vis[['CIG_before', 'CIG_after']] *= 1 / 0.3048
        cig_range = [
            0,
            15,
            40,
            75,
            125,
            175,
            225,
            275,
            325,
            375,
            450,
            550,
            650,
            750,
            850,
            950,
            1050,
            1150,
            1250,
            1350,
            1450,
            1550,
            1650,
            1750,
            1850,
            1950,
            2050,
            2150,
            2250,
            2350,
            2450,
            2550,
            2650,
            2750,
            2850,
            2950,
            3250,
            3750,
            4500,
            100000
        ]

        cig_values = [
            0,
            30,
            50,
            100,
            150,
            200,
            250,
            300,
            350,
            400,
            500,
            600,
            700,
            800,
            900,
            1000,
            1100,
            1200,
            1300,
            1400,
            1500,
            1600,
            1700,
            1800,
            1900,
            2000,
            2100,
            2200,
            2300,
            2400,
            2500,
            2600,
            2700,
            2800,
            2900,
            3000,
            3500,
            4000,
            ''
        ]

        i = 0
        while True:
            idx = np.where((vis['CIG_before'] > cig_range[i]) & (vis['CIG_before'] <= cig_range[i + 1]))[0]
            idx = vis.index[idx]
            vis.loc[idx, 'CIG_before'] = cig_values[i]
            i += 1
            if i == len(cig_values):
                break

        i = 0
        while True:
            idx = np.where((vis['CIG_after'] > cig_range[i]) & (vis['CIG_after'] <= cig_range[i + 1]))[0]
            idx = vis.index[idx]
            vis.loc[idx, 'CIG_after'] = cig_values[i]
            i += 1
            if i == len(cig_values):
                break

        vis['WNDSPD_before'] *= 1 / 0.514444
        vis = vis.round({'WNDSPD_before': 0})
        vis = vis.round({'WNDDIR_before': -1})

        wx_list = [
            '',
            '',
            'RA',
            'SNRA',
            'SN',
            'SNRA',
            '',
            '',
            ''
        ]
        wx_telop = []
        for idx in vis.index:
            if idx in df_before_06_23.index:
                wx_prob = df_before_06_23.loc[idx, [
                    'WX_telop_100',
                    'WX_telop_200',
                    'WX_telop_300',
                    'WX_telop_340',
                    'WX_telop_400',
                    'WX_telop_430',
                    'WX_telop_500',
                    'WX_telop_600',
                    'WX_telop_610'
                ]]

                prc = df_before_06_23.loc[idx, 'PRCRIN_1HOUR_TOTAL']
                if (prc >= 5.) and (prc < 10.):
                    wx = ''
                elif prc >= 10.:
                    wx = '+'
                else:
                    wx = '-'
                tnd = df_before_06_23.loc[idx, 'TNDSTM_prob']
                if tnd >= 50.:
                    wx += 'TS'
                wx += wx_list[int(np.argmax(wx_prob.values))]

                v = df_before_06_23.loc[idx, 'VIS']
                tmpr = df_before_06_23.loc[idx, 'AIRTMP']
                if v < 1000.:
                    wx += ' FG'
                elif (v < 1000.) and (tmpr < 0.):
                    wx += ' FZFG'
                elif (v >= 1000.) and (v <= 5000.):
                    wx += ' BR'

                wx_telop.append(wx)
        wx_telop = pd.DataFrame(wx_telop, index=df_before_06_23.index, columns=['WX_before'])
        vis = pd.concat([vis, wx_telop], axis=1)
        vis['WX_after'] = vis['WX_after'].str.replace(' ', '')
        vis['WX_after'] = vis['WX_after'].str.replace('_', '-')
        vis.dropna(inplace=True)

        vis_edit = np.where(vis['VIS_before'] != vis['VIS_after'], '*', '')
        cig_edit = np.where(vis['CIG_before'] != vis['CIG_after'], '*', '')
        wspd_edit = np.where(vis['WNDSPD_before'] != vis['WNDSPD_after'], '*', '')
        wdir_edit = np.where(vis['WNDDIR_before'] != vis['WNDDIR_after'], '*', '')
        wx_edit = np.where(vis['WX_before'] != vis['WX_after'], '*', '')

        vis_edit_rate = calc_edit_rate(vis_edit)
        cig_edit_rate = calc_edit_rate(cig_edit)
        wspd_edit_rate = calc_edit_rate(wspd_edit)
        wdir_edit_rate = calc_edit_rate(wdir_edit)
        wx_edit_rate = calc_edit_rate(wx_edit)

        vis['VIS edit'] = vis_edit
        vis['CIG edit'] = cig_edit
        vis['WNDSPD edit'] = wspd_edit
        vis['WDIR edit'] = wdir_edit
        vis['WX edit'] = wx_edit

        # 時系列データが欲しい場合は書き足す
        columns = ['ICAO', 'date', 'VIS_before', 'VIS_after', 'VIS edit', 'CIG_before', 'CIG_after', 'CIG edit',
                   'WNDSPD_before', 'WNDSPD_after', 'WNDSPD edit', 'WNDDIR_before', 'WNDDIR_after', 'WDIR edit',
                   'WX_before', 'WX_after', 'WX edit']
        vis = vis[columns]

        df_edit = pd.DataFrame([[
            icao,
            len(vis_edit),
            len(vis_edit[vis_edit == '*']),
            vis_edit_rate,
            len(cig_edit[cig_edit == '*']),
            cig_edit_rate,
            len(wspd_edit[wspd_edit == '*']),
            wspd_edit_rate,
            len(wdir_edit[wdir_edit == '*']),
            wdir_edit_rate,
            len(wx_edit[wx_edit == '*']),
            wx_edit_rate
        ]], columns=['ICAO', 'All', 'Vis edit', 'VIS edit rate', 'CIG edit', 'CIG edit rate',
                     'WNDSPD edit', 'WNDSPD edit rate', 'WDIR edit', 'WDIR edit rate',
                     'WX edit', 'WX edit rate'])

        df_edit.to_csv('%s/edit_rate_%s_06_23.csv' % (save_dir, icao), index=False)
        df_edit_all = pd.concat([df_edit_all, df_edit])

    df_edit_all = df_edit_all.round(3)
    df_edit_all.to_csv('%s/edit_rate_all_06_23.csv' % save_dir, index=False)


def main():
    # edit_rate_00_05()
    edit_rate_06_23()


if __name__ == '__main__':
    main()

'''
memo
[
'AREA', 'ICAO', 'date', 'PRCRIN_1HOUR_TOTAL', 'PRCRIN_1HOUR_TOTAL_bure', 'VIS', 'VIS_bure', 'CLING', 'CLING_bure', 
'AIRTMP', 'AIRTMP_bure', 'WNDSPD', 'WNDSPD_bure', 'WNDDIR', 'GUSTS', 'CAPE', 'CAPE_bure', 'CIN', 'SSI', 'TNDSTM_prob', 
'WX_telop_100', 'WX_telop_200', 'WX_telop_300', 'WX_telop_340', 'WX_telop_400', 'WX_telop_430', 'WX_telop_500', 
'WX_telop_600', 'WX_telop_610'
]
'''
