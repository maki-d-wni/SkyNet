import re
import os
import shutil

import pandas as pd


def msm_file_arrangement(file_path, output_path):
    list_dir = os.listdir(file_path)
    list_dir.sort()
    for d in list_dir:
        if os.path.isdir(file_path + "/" + d):
            msm_file_arrangement(file_path + "/" + d, output_path)
        else:
            if os.path.isfile(output_path + "/" + d):
                print(d, "already exist.")
            else:
                print(d)
                shutil.copyfile(file_path + "/" + d, output_path + "/" + d)
    return


def main():
    msm_info = {
        'surface': {
            '400220000': {
                'base time': '00',
                'validity time': '0015'
            },
            '400220001': {
                'base time': '03',
                'validity time': '0015'
            },
            '400220002': {
                'base time': '06',
                'validity time': '0015'
            },
            '400220003': {
                'base time': '09',
                'validity time': '0015'
            },
            '400220004': {
                'base time': '12',
                'validity time': '0015'
            },
            '400220005': {
                'base time': '15',
                'validity time': '0015'
            },
            '400220006': {
                'base time': '18',
                'validity time': '0015'
            },
            '400220007': {
                'base time': '21',
                'validity time': '0015'
            },
            '400220115': {
                'base time': '03',
                'validity time': '1633'
            }
        },
        'upper': {
            '400220008': {
                'base time': '00',
                'validity time': '0015'
            },
            '400220009': {
                'base time': '03',
                'validity time': '0015'
            },
            '400220010': {
                'base time': '06',
                'validity time': '0015'
            },
            '400220011': {
                'base time': '09',
                'validity time': '0015'
            },
            '400220012': {
                'base time': '12',
                'validity time': '0015'
            },
            '400220013': {
                'base time': '15',
                'validity time': '0015'
            },
            '400220014': {
                'base time': '18',
                'validity time': '0015'
            },
            '400220015': {
                'base time': '21',
                'validity time': '0015'
            },
            '400220123': {
                'base time': '03',
                'validity time': '1633'
            }
        }
    }

    gsm_info = {
        'surface': {
            '400220143': {
                'base time': '12',
                'validity time': '0084'
            }
        },
        'upper': {
            '400220148': {
                'base time': '12',
                'validity time': '0084'
            }
        }
    }

    nwp_type = 'MSM'
    layer = 'upper'
    tagid = '400220123'
    # move_nwp_file(nwp_type, nwp_info=msm_info)

    for bt in ['03']:
        input_path = '/home/ai-corner/part1/%s/%s' % (nwp_type, tagid)
        date_dirs = os.listdir(input_path)
        for d in date_dirs:
            output_path = '/home/ai-corner/part1/%s/%s/bt%s/vt1633/%s' % (nwp_type, layer, bt, d)
            # os.system('mkdir %s' % output_path)
            # os.system('mv %s/%s %s/' % (input_path, d, output_path))
            # print('mkdir %s' % output_path)
            print('mv %s/%s %s/' % (input_path, d, output_path))


if __name__ == '__main__':
    main()
