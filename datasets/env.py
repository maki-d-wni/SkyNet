import re
import os


def move_nwp_file(data_path, nwp_info):
    pattern = re.compile(r'\d{8}_\d{6}.\d{3}')
    for layer in nwp_info:
        for tag_id in nwp_info[layer]:
            bt = nwp_info[layer][tag_id]['base time']
            vt = nwp_info[layer][tag_id]['validity time']
            file_path = '%s/%s' % (data_path, tag_id)
            output_path = '%s/%s/bt%s/vt%s' % (data_path, layer, bt, vt)
            __move_nwp_file(file_path, output_path, pattern)


def __move_nwp_file(file_path, output_path, pattern):
    list_dir = os.listdir(file_path)
    list_dir.sort()
    for d in list_dir:
        if os.path.isdir(file_path + "/" + d):
            if pattern.match(d):
                print('mv %s/%s %s' % (file_path, d, output_path))
                os.system('mv %s/%s %s/%s' % (file_path, d, output_path, d))
            else:
                __move_nwp_file(file_path + "/" + d, output_path, pattern)
        else:
            if not pattern.match(d):
                print('remove %s' % d)
                os.system('rm \'%s/%s\'' % (file_path, d))


def convert_dict_construction(old, new: dict, pwd: str, depth: int):
    new = __apply_convert_dict_construction(old, new, pwd)
    keys = list(new.keys())
    skeys = [key.split(pwd) for key in keys]
    tn = []
    for i, d in enumerate(skeys):
        if len(d) > 2:
            tn.append(len(d) - 1 - depth)
        else:
            tn.append(1)

    keys = ["/".join(key[d:]) for d, key in zip(tn, skeys)]

    new = {key: new[n] for key, n in zip(keys, new)}

    return new


def __apply_convert_dict_construction(old, new: dict, pwd: str):
    if type(old) == dict:
        for o in old:
            nkey = pwd
            if type(o) == str:
                if pwd == "/":
                    nkey += o
                else:
                    nkey += "/" + o
            if __check_iterable(old[o]):
                __apply_convert_dict_construction(old[o], new, nkey)
            else:
                if nkey in new.keys():
                    new[nkey].append(old[o])
                else:
                    new[nkey] = [old[o]]
    else:
        for o in old:
            nkey = pwd
            if type(o) == str:
                if pwd == "/":
                    nkey += o
                else:
                    nkey += "/" + o
            if __check_iterable(o):
                __apply_convert_dict_construction(o, new, nkey)

    return new


def __check_iterable(obj):
    if hasattr(obj, "__iter__"):
        if type(obj) == str:
            return False
        else:
            return True
    else:
        return False
