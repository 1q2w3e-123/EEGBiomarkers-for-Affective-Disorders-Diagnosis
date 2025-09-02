import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import mne
import glob
import math
import os
import pywt
import matplotlib.pyplot as plt
import matplotlib
# import pandas as pd
# from generate_data import generate
from torchvision import transforms
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lyt1.model import Lin
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
yu=100
yu_num=20
save_sz='/data0/lyt/data/good/RestEyesOpen/BD_npy'
save_spec='/data0/lyt/data/npy/open/HC_spec'
channels=['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'O1', 'OZ', 'O2']
s=''
bad_p=[]
# with open('/data0/lyt/model_check/test_bad.txt','r') as f:
#     s=f.read()
#     # bad_p=s.split('\n')
#     bad_p=eval(s)
#     print(bad_p,type(bad_p))
def compute_power(raw):
    FREQ_BANDS = {"delta": [0.5, 4],
                  "theta": [4, 8],
                  "alpha": [8, 14],
                  "beta": [14, 30],
                  "gama": [30, 50]}
    feature_vector = {}
    # 遍历每个频段
    for band in FREQ_BANDS:
        # 提取每个频段的数据，不打印信息
        # raw_band = raw.copy().filter(l_freq=FREQ_BANDS[band][0], h_freq=FREQ_BANDS[band][1], verbose=False)
        # # 计算能量
        # power = np.sum(raw_band.get_data() ** 2, axis=1) / raw_band.n_times
        # 添加到特征向量
        # feature_vector.append(power)

        power = raw.compute_psd(picks='all', method='welch', fmin=FREQ_BANDS[band][0],
                                           fmax=FREQ_BANDS[band][1], verbose=False)
        # print(power.shape,type(power))
        # for i in range(power.shape[0]):
        x=power.get_data()
        feature_vector[band]=x
    return feature_vector

def interpolate(raw):
    raw_data=[]
    raw_spect=[]
    tmin=10.0
    tmax=12.0-1/250
    while tmax<len(raw)/250:
        tem=raw.copy()
        raw1=tem.crop(tmin,tmax)
        raw_d=raw1.get_data()
        # bad=['HEO','VEO']
        bad=[]
        for k,channel_data in enumerate(raw_d):
            if k>=60: #EOG不处理
                continue
            max_num,min_num = 0,0
            for i in channel_data:
                if np.abs(i)>100e-6:
                    max_num+=1
                elif np.abs(i)<1e-6:
                    min_num += 1
            if max_num>=yu_num or min_num>=100:
                bad.append(channels[k])
        if len(bad)<10:
            raw1.info["bads"]=bad
            raw1.interpolate_bads(exclude=['HEO','VEO']) #不插值EOG

            #提取频带
            power = compute_power(raw1)
            # print(power.shape)
            # assert(1==0)
            #结束

            raw1_data=raw1.get_data()
            raw1_data=np.expand_dims(raw1_data,axis=0)
            raw_spect.append(power)
            if len(raw_data)==0:
                raw_data=raw1_data
            else:
                raw_data=np.append(raw_data,raw1_data,axis=-1)
        tmin+=1
        tmax+=1
    data=np.array([])
    if len(raw_data)>0:
        raw_data = np.squeeze(raw_data)#.transpose() #shape: (60+2)*(n_samples*500)
        scaler = StandardScaler()
        scaler.fit(raw_data[:60])  # 对每个个体的EEG通道做标准化处理 以timestamp为单位 EOG不处理
        raw_data[:60] = scaler.transform(raw_data[:60])
        #raw_data = raw_data.transpose()
    # print(type(raw_data),len(raw_data),len(raw_data[0]))
        
        length=int(raw_data.shape[1]/500)
        data=raw_data[:,:500]
        data=np.expand_dims(data,axis=0)
        for i in range(1,length):
            tem=raw_data[:,i*500:(i+1)*500]
            tem=np.expand_dims(tem,axis=0)
            data=np.append(data,tem,axis=0)
    # print(data.shape)
    # assert(1==0)
    return data,raw_spect  #shape: (60+2)*(n_samples*500)

def train_test(data_list,ratio,pos):
    length=int(len(data_list)*ratio)
    start=int(len(data_list)*pos)
    # print(start,length)
    test_list=data_list[start:start+length]
    train_list=data_list[:start]
    train_list.extend(data_list[start+length:])
    val_len=int(ratio*len(data_list))
    val_list=train_list[:val_len]
    train_list=train_list[val_len:]
    return train_list,test_list,val_list

def generate_epo(list,kind):
    data={}
    for i in tqdm(range(len(list)),desc='get_data:'):
        raw = mne.io.read_raw(list[i],preload=True)
        # compute_power(raw)
        # assert(1==0)
        epoch_data,spec=interpolate(raw)
        # print(len(spec))
        np_spec=np.array(spec)
        # print(np_spec.shape)
        # print(np_spec[0])
        # print(np_spec[0]['alpha'].shape)
        # assert(1==0)
        # print(epoch_data.shape,spec.shape)
        # assert(1==0)
        # with open('sz.txt','a') as f:
        #     f.write(list[i].split('/')[-1]+':'+str(epoch_data.shape)+'\n')
        if epoch_data.shape[0]>50:
            name=list[i].split('/')[-1]
            name=name[:-3]+'npy'
            # spec_name=name[:-4]+'_spec.npy'
            # print(save_sz+'/'+name,epoch_data.shape)
            # assert(1==0)
            np.save(save_sz+'/'+name,epoch_data)
            # np.save(save_spec+'/'+spec_name,np_spec)
            data[list[i]]=epoch_data
    return data

def getdata(path_list,data_dict):
    data=[]
    for i in path_list:
        name=i.split('/')[-1]
        name=name[:-3]+'npy'
        tem=np.load(save_sz+'/'+name)

        # length=int(tem.shape[1]/500)
        # tem_data=tem[:,:500]
        # tem_data=np.expand_dims(tem_data,axis=0)
        # for i in range(1,length):
        #     tem1=tem[:,i*500:(i+1)*500]
        #     tem1=np.expand_dims(tem1,axis=0)
        #     tem_data=np.append(tem_data,tem1,axis=0)

        if len(data)==0:
            data=tem
        else:
            data=np.append(data,tem,axis=0)
    return data

open_sz='/data0/lyt/data/good/RestEyesOpen/BD'
# open_hc='/data0/lyt/data/0.7ICA/RestEyesOpen/HC'
# open_bd='/data0/violin/projects/szbd_practice/64_rest_BD_good_ICA'
# open_sad='/data0/violin/projects/szbd_practice/64_rest_SAD_good_ICA'
save_dir='/data0/lyt/data/good/RestEyesOpen/npy'
sz_list=os.listdir(open_sz)
# hc_list=os.listdir(open_hc)
# bd_all=os.listdir(open_bd)
# bd_path=[]
# for i in bd_all:
#     s=i.split('-')
#     # print(s[-3])
#     # assert(1==0)
#     if s[-3]=='RestEyesOpen':
#         bd_path.append(open_bd+'/'+i)
# sad_all=os.listdir(open_sad)
# sad_path=[]
# for i in sad_all:
#     s=i.split('-')
#     if s[-3]=='RestEyesOpen':
#         sad_path.append(open_sad+'/'+i)
sz_path=[open_sz+'/'+i for i in sz_list]
# hc_path=[open_hc+'/'+i for i in hc_list]
sz_all_data={}
sz_all_data=generate_epo(sz_path,'sz')
# print('data_dict: ',len(sz_all_data))
# sz_path=[save_sz+'/'+i for i in os.listdir(save_sz)]
# assert(1==0)
sz_path=[]
print('before:',len(os.listdir(save_sz)))
for i in os.listdir(save_sz):
    name=save_sz+'/'+i
    if not i in bad_p:
        sz_path.append(name)
print('all:',len(sz_path))
for i in range(5):
    split_point=i/5.0
    train_sz,test_sz,val_sz=train_test(sz_path,0.2,split_point)
    # train_hc,test_hc,val_hc=train_test(hc_path,0.2,split_point)
    train_sz_data=getdata(train_sz,sz_all_data)
    np.save(save_dir+'/train_sz_test_{}_{}_{}.npy'.format(yu,yu_num,split_point),train_sz_data)
    test_sz_data=getdata(test_sz,sz_all_data)
    np.save(save_dir+'/test_sz_test_{}_{}_{}.npy'.format(yu,yu_num,split_point),test_sz_data)
    val_sz_data=getdata(val_sz,sz_all_data)
    np.save(save_dir+'/val_sz_test_{}_{}_{}.npy'.format(yu,yu_num,split_point),val_sz_data)
    print(train_sz_data.shape,test_sz_data.shape,val_sz_data.shape)
    print(len(train_sz),len(test_sz),len(val_sz))

