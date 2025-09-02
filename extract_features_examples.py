import os
import scipy.io as sio
from main_preprocessing import Preprocessing
import mne
import pandas as pd
import lmdb
import pickle
import numpy as np
import antropy as ant
from mi import compute_mi
from statsmodels.tsa.stattools import grangercausalitytests
from tqdm import tqdm
from connectivity import plv_connectivity,pli_connectivity,ccf_connectivity,coh_connectivity,icoh_connectivity
from mne_connectivity import spectral_connectivity_epochs
from torcheeg import transforms
from extract_feature import extract_connectivity,data_append
from lds import smooth_feature
import warnings
import mne_microstates
from mne_connectivity import spectral_connectivity_epochs,vector_auto_regression
warnings.filterwarnings('ignore')
# %%
root_dir = '/data0/cyn/DEAP/data_preprocessed_python'
files = [file for file in os.listdir(root_dir)]
files = sorted(files,key=lambda x: (int(x[1:3])))


files_dict = {
    'train':files[:32]
    # 'val':files[20:21],
    # 'test':files[20:21],
}


eeg_duration = 1
baseline_duration = 3  # 秒

all_fre ={"theta": [4, 8],
                  "alpha": [8, 14],
                  "beta": [14, 30],
                  "gama": [30, 50]}

t1 = transforms.BandSampleEntropy(band_dict=all_fre,sampling_rate=128,R=0.2)
t2 = transforms.BandHiguchiFractalDimension(band_dict=all_fre,sampling_rate=128)
t3 = transforms.BandHjorth(band_dict=all_fre,sampling_rate=128,mode='mobility')
t4 = transforms.BandHjorth(band_dict=all_fre,sampling_rate=128,mode='complexity')
t = transforms.BandDetrendedFluctuationAnalysis(band_dict=all_fre,sampling_rate=128)


######################################################################
# extract features method
######################################################################
    # "AppEn":transforms.BandApproximateEntropy(band_dict=all_fre,sampling_rate=128),
    # "SamEn":transforms.BandSampleEntropy(band_dict=all_fre,sampling_rate=128,R=0.2),
    # "HFD":transforms.BandHiguchiFractalDimension(band_dict=all_fre,sampling_rate=128),
    # "DFA":transforms.BandDetrendedFluctuationAnalysis(band_dict=all_fre,sampling_rate=128),
    # "hjc":transforms.BandHjorth(band_dict=all_fre,sampling_rate=128,mode='complexity'),
    # "hjm":transforms.BandHjorth(band_dict=all_fre,sampling_rate=128,mode='mobility'),
    # "DE":transforms.BandDifferentialEntropy(band_dict=all_fre),
    # "micro_state":mne_microstates.segment(one, n_states=4,
    #                                                         random_state=0,
    #                                                         return_polarity=True),
    # "GC":compute_gc_matrix,
    # "PLV":extract_connectivity(data,"plv"),
    # "PLI":extract_connectivity(data,"pli"),
    # "cohy":spectral_connectivity_epochs(data=sample1,method='cohy',sfreq=128,fmin=4,fmax=50,mode='multitaper',faverage=True),
    # "imcoh":spectral_connectivity_epochs(data=sample1,method='imcoh',sfreq=128,fmin=4,fmax=50,mode='multitaper',faverage=True),
    # "dpli":spectral_connectivity_epochs(data=sample1,method='dpli',sfreq=128,fmin=4,fmax=50,mode='multitaper',faverage=True),
    # "SpecEn":transforms.BandSampleEntropy(band_dict=all_fre,sampling_rate=128),
    # "kur":transforms.BandKurtosis(),
    # "ske":transforms.BandSkewness(),
    # "PSD":mne.time_frequency.psd_array_multitaper(sample,128,fmin=4,fmax=50),
    # "PermEn":ant.perm_entropy(),
    # "MI":get_MI(),
    # "PFD":transforms.BandPetrosianFractalDimension(band_dict=all_fre,sampling_rate=128),
    # "SVDEn":transforms.BandSVDEntropy(band_dict=all_fre,sampling_rate=128),
    # "CCF":extract_connectivity(data,"ccf"),,
    # "coh":spectral_connectivity_epochs(data=sample1,method='coh',sfreq=128,fmin=4,fmax=50,mode='multitaper',faverage=True)
def compute_gc_matrix(data, maxlag=2, verbose=False):
    """
    计算 Granger 因果矩阵 (num_channels x num_channels)

    参数:
    - data: ndarray, shape (num_channels, n_times)
    - maxlag: int, VAR模型的最大滞后阶数
    - verbose: 是否打印grangercausalitytests的详细结果

    返回:
    - gc_matrix: ndarray, shape (num_channels, num_channels)
                 gc_matrix[i, j] 表示 通道i → 通道j 的GC强度
    """
    num_channels, n_times = data.shape
    gc_matrix = np.zeros((num_channels, num_channels))

    for i in range(num_channels):
        for j in range(num_channels):
            if i == j:
                continue  # 自因果设为0
            # 构造输入格式 (T, 2)，第0列是被预测的序列，第1列是潜在的因果序列
            test_data = np.vstack([data[j], data[i]]).T  

            try:
                results = grangercausalitytests(test_data, maxlag=maxlag, verbose=verbose)
                # 提取某个指标作为 GC 强度，这里用 maxlag 阶的 F 检验的 p-value
                p_value = results[maxlag][0]['ssr_ftest'][1]
                gc_matrix[i, j] = -np.log(p_value + 1e-10)  # p 越小，因果越强（取负log方便可视化）
            except Exception as e:
                print(f"GC计算失败: {i}->{j}, 错误: {e}")
                gc_matrix[i, j] = 0

    return gc_matrix

def get_MI(data):
    target=[]
    for i in range(40):
        target.append([0 for i in range(40)])
    for i in range(40):
        for j in range(i,40):
            mi = compute_mi(data[i], data[j],20)
            target[i][j]=mi
            target[j][i]=mi
    return np.array(target)

all_de=[]
all_plv=[]
all_lds_de=[]
all_lds_plv=[]
all_label=[]
all_data=[]
num=0
# path = '/data0/violin/projects/szbd_practice/szbd_dataset_eeg_64_noICA'
# files = os.listdir(path)
# dict_num = {}
# for file in files:
#     s = file.split('-')
#     if s[-3]!='RestEyesOpen' and s[-3]!='RestEyesClosed':
#         continue
#     name = s[-3]+"_"+s[-2]
#     if not name in dict_num.keys():
#         dict_num[name]=1
#     else:
#         dict_num[name]+=1
# print(dict_num)
# assert(1==0)
for files_key in files_dict.keys():
    for file in tqdm(files_dict[files_key]):
        # num+=1
        # if num<=9:
        #     continue
        # print(files_dict[files_key])
        # assert(1==0)
        # print(file[:-4])
        # assert(1==0)
        data_path = os.path.join(root_dir, file)
        data_label = pickle.load(open(data_path, 'rb'), encoding='latin1')
        data = data_label['data']
        data = data.reshape(40,40,63,128)
        data = data.transpose(0, 2, 1, 3)
        label_valence = data_label['labels']
        baseline_values = np.mean(data[:, :baseline_duration, :, :], axis=1)  # shape (40, 40, 128)
        data = data - baseline_values[:, np.newaxis, :, :]
        data = data[:,baseline_duration:]
        # data = data.transpose(0,2,1,3)
        # print(data.shape)
        # assert(1==0)
        sub_de=[]
        sub_plv=[]
        de_lds=[]
        plv_lds=[]
        sub_label=[]
        sub_data=[]
        for i in tqdm(range(data.shape[0])):
            samples = data[i]
            # if label_valence[i]>5:
            #     label_valence[i]=0
            # else:
            #     label_valence[i]=1
            session_de=[]
            session_plv=[]
            session_label=[]
            session_data=[]
            # print(data.shape[1],'fffffffff')
            # assert(1==0)
            for j in range(data.shape[1] // eeg_duration):
                sample = samples[eeg_duration * j:eeg_duration * (j + 1)][0]
                maps = []
                for x1 in sample:
                    tem = ant.perm_entropy(x1, normalize=True)
                    maps.append(tem)
                maps = np.array(maps)
                ske = np.expand_dims(maps,axis=-1)
                # print(maps.shape)
                # assert(1==0)
                # plv = extract_connectivity(sample,'pli')
                session_de=data_append(session_de,ske)
                # session_plv=data_append(session_plv,maps3)
                # session_label=data_append(session_label,label_valence[i])
                # session_data=data_append(session_data,maps2)
            # tem_plv=LDS_plv(session_plv)
            # tem_de=session_de
            # tem_de=tem_de.transpose(0, 2, 1)
            # tem_de=smooth_feature(tem_de)
            # tem_de=tem_de.transpose(0, 2, 1)
            sub_de=data_append(sub_de,session_de)
            # sub_plv=data_append(sub_plv,session_plv)
            # de_lds=data_append(de_lds,tem_de)
            # plv_lds=data_append(plv_lds,tem_plv)
            # sub_label=data_append(sub_label,session_label)
            # sub_data=data_append(sub_data,session_data)
            # print(sub_de.shape,de_lds.shape)
            # assert(1==0)
        # all_lds_de=data_append(all_lds_de,de_lds)
        # all_lds_plv=data_append(all_lds_plv,plv_lds)
        # np.save('/data0/lyt/emotion/seed/DEAP/gc_{}.npy'.format(file[:-4]),all_de)

        all_de=data_append(all_de,sub_de)
        # all_plv=data_append(all_plv,sub_plv)
        # all_label=data_append(all_label,sub_label)
        # all_data=data_append(all_data,sub_data)
    # np.save('/data0/lyt/emotion/seed/DEAP/plvLDS.npy',all_lds_plv)
    # np.save('/data0/lyt/emotion/seed/DEAP/deLDS4band.npy',all_lds_de)
    # np.save('/data0/lyt/emotion/seed/DEAP/dpli.npy',all_plv)
    np.save('/data0/lyt/emotion/seed/DEAP/permen.npy',all_de)
    # np.save('/data0/lyt/emotion/seed/DEAP/label.npy',all_label)
    # np.save('/data0/lyt/emotion/seed/DEAP/imcoh.npy',all_data)
    print(all_de.shape)