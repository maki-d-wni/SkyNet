#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from multiprocessing import Pool
import multiprocessing as multi
from datetime import datetime
import sys
import os
from pathlib import Path
import csv
import re

from skynet.live.vis import Vis_Pred


def One_Process_Vis_Pred(i, args_set):
    myname = sys.argv[0]

    for j, args in args_set.iterrows():
        Vis_Pred(args['MODEL'], args['CONTXT'], args['LCLID'], args['test_dir'], args['input_dir'], args['fit_dir'],
                 args['pred_dir'], args['errfile'])
    # MODEL -> JMA_MSM
    # CONTXT -> GLOBAL_METAR
    # LCLID  -> RJAA
    # test_dir -> /home/data/point_data/$MODEL/$YYYY/$MM/$DD/$HH : input csv
    # input_dir -> /home/data/ARC-common/fit_input/$MODEL/vis : training data csv
    # fit_dir  -> /home/data/ARC-common/fit_output/$MODEL/vis : input pkl
    # pred_dir -> /home/data/ARC-pred/pred_output/$MODEL/$YYYY/$MM/$DD/$HH/vis : output csv


def main():
    # --- start time
    start = datetime.now()
    print("Python Start Time:{0}".format(start))

    # --- Argument
    args = sys.argv
    myname = args[0]
    if len(args) != 9 and len(args) != 10:
        print(
            "usage: {:s} [model name] [data dir] [Points List(with header)] [num of process] [base year] [base month] [base day] [base hour] ([region or ens])".format(
                args[0]))
        sys.exit()
    model = args[1]  # JMA_MSM
    data_dir = args[2]  # /home/data
    station_list = args[3]  # /usr/compass/common/tbl.v1/ARC.JP_with_header.tbl.new
    Nproc = int(args[4])  # 1( ~ 8)
    byyyy = int(args[5])  # YYYYY
    bmm = int(args[6])  # MM
    bdd = int(args[7])  # DD
    bhh = int(args[8])  # HH

    region = ""
    arcmodel = model

    if len(args) == 10:
        region = args[9]
        if str(model) in region and (re.match(model + '_[0-9]{2}', region) or re.match(model + '_CTRL', region)):
            arcmodel = model
        else:
            arcmodel = region

    Njobs = Nproc * 1

    # --- Set Directory
    mydir = os.path.dirname(os.path.abspath(__file__))
    # mydir     = "/usr/compass/ARC-pred"
    test_dir = "{:s}/point_data/{:s}/{:s}/{:04d}/{:02d}/{:02d}/{:02d}".format(data_dir, model, region, byyyy, bmm, bdd,
                                                                              bhh)
    input_dir = "{:s}/ARC-common/fit_input/{:s}/vis".format(data_dir, arcmodel)
    fit_dir = "{:s}/ARC-common/fit_output/{:s}/vis".format(data_dir, arcmodel)
    pred_dir = "{:s}/ARC-pred/pred_output/{:s}/{:s}/{:04d}/{:02d}/{:02d}/{:02d}/vis".format(data_dir, model, region,
                                                                                            byyyy, bmm, bdd, bhh)

    # --- Set Error file
    errfile = "{:s}/ARC-pred/tmp/pred.{:s}.error".format(data_dir, arcmodel)

    # --- Read Point List
    if not os.path.exists(station_list):
        print("{:s}: [Error] {:s} is not found !".format(myname, station_list))
        Path(errfile).touch()
        sys.exit()
    all_stations = pd.read_csv(station_list, sep=",", dtype='str')

    '''
#--- Read Need List
elemfile = "{:s}/tbl/dataset_tbl/train_element.{:s}.vis.txt" .format(mydir,arcmodel)
if not os.path.exists(elemfile):
    print ("{:s}: [Error] {:s} is not found !" .format(myname,elemfile))
    Path(errfile).touch()
    sys.exit()
Need_Elems = ['YEAR','MON','DAY','HOUR']
f = open(elemfile)
for line in f:
    if line.find('HEAD') > -1:
        continue
    dat    = line.rstrip().split("\t")
    elem   = dat[0]
    levels = dat[1].split(",")
    for level in levels:
        model_elem = "{:s}-{:s}" .format(level,elem)
        Need_Elems = Need_Elems + [model_elem]
f.close()
print (Need_Elems)
    '''

    # --- Make Directory
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    # --- Make All Arguments
    Models = [arcmodel for i in range(len(all_stations))]
    Test_Dirs = [test_dir for i in range(len(all_stations))]
    Input_Dirs = [input_dir for i in range(len(all_stations))]
    Fit_Dirs = [fit_dir for i in range(len(all_stations))]
    Pred_Dirs = [pred_dir for i in range(len(all_stations))]
    Errfiles = [errfile for i in range(len(all_stations))]
    # Need_Elems_List = [Need_Elems for i in range(len(all_stations))]

    all_args = pd.DataFrame(
        {
            'MODEL': Models,
            'CONTXT': all_stations['CONTXT'].astype('str'),
            'LCLID': all_stations['LCLID'].astype('str'),
            'test_dir': Test_Dirs,
            'input_dir': Input_Dirs,
            'fit_dir': Fit_Dirs,
            'pred_dir': Pred_Dirs,
            'errfile': Errfiles,
        }
    )

    # --- Make Split Arguments & Split Set for Multi Process
    SplitArgs = []
    SplitSet = []

    if len(all_args) < Njobs:
        print("{:s}: [Error] num of process is bigger than list size !".format(myname))
        Path(errfile).touch()
        sys.exit()

    stride = len(all_args) // Njobs
    remain = len(all_args) % Njobs

    # --  Make Split Stations
    for i in range(0, Njobs):
        start = stride * (i)
        end = stride * (i + 1)
        if i == (Njobs - 1):
            end = end + remain
        SplitArgs.append(all_args[start:end])

    # --  Make Split Set
    for i in range(0, Njobs):
        tuple = (i, SplitArgs[i])
        SplitSet.append(tuple)

    # --- Multi Process
    with Pool(Njobs) as p:
        p.starmap(One_Process_Vis_Pred, SplitSet)

    # --- Finish Time
    end = datetime.now()
    print("{0}: [Completed] Python Finish Time:{1}".format(myname, end))


if __name__ == "__main__":
    main()
