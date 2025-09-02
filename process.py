import csv
import mne
import os
import matplotlib.pyplot as plt
import shutil 
from tqdm import tqdm
from mne_icalabel import label_components
from mne.preprocessing import ICA
import numpy as np
filename = '/data0/lyt/nimh/SAD_EO.csv'
def process(d):
    name=d[0]
    bad_channel=eval(d[1])
    path='/data0/zrz/szbd_data/raw/RestEyesOpen/'+name
    raw = mne.io.read_raw_cnt(path,preload=True)
    if len(raw.info['ch_names'])<66:
        return
    raw = raw.resample(250)
    raw.drop_channels(raw.info['ch_names'][66:])
    raw.set_eeg_reference(ref_channels=['M1','M2'])
    raw.drop_channels(['CB1','CB2','M1','M2'])
    raw.notch_filter(60,fir_design='firwin')
    raw = raw.filter(0.1,50.0,fir_design='firwin')
    raw.info["bads"]=bad_channel
    raw.interpolate_bads()
    
    raw.save('/data0/lyt/data/good/RestEyesOpen/SAD/'+name[:-3]+'fif',overwrite=True)
    return

with open(filename, "r") as csvfile:
    content=csv.reader(csvfile)
    for i,row in enumerate(content):
        if i==0:
            continue
        process(row)