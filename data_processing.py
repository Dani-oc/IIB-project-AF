## IMPORT MODULES
import numpy as np
import scipy
import scipy.signal as signal
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import pywt
import wfdb
import math
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm
from skimage.restoration import denoise_wavelet


tags_ecg_feas1 = pd.read_csv(r"Data\\feas1\\rec_data_anon.csv")
tags_pt_feas1 = pd.read_csv(r"Data\\feas1\\pt_data_anon.csv")
key_1 = {'AF':2, 'maybeAF':3, 'noAF':4, 'undecided':6}
tags_ecg_feas2 = pd.read_csv(r"Data\\feas2\\rec_data_anon.csv")
tags_pt_feas2 = pd.read_csv(r"Data\\feas2\\pt_data_anon.csv")
key_2 = {'AF':1, 'maybeAF':2, 'noAF':3, 'noisy':4, 'undecided':6}
tags_ecg_trial = pd.read_csv(r"Data\\trial\\rec_data_anon.csv")
tags_pt_trial = pd.read_csv(r"Data\\trial\\pt_data_anon.csv")
key_t = {'AF':1, 'maybeAF':2, 'noAF':3, 'other':4, 'noisy':5, 'undecided':6}

class ecg:

    def __init__(self, tags_ecg_feas1 = tags_ecg_feas1, tags_pt_feas1 = tags_pt_feas1, 
                 tags_ecg_feas2 = tags_ecg_feas2, tags_pt_feas2 = tags_pt_feas2, 
                 key_1 = key_1, key_2 = key_2):
        self.tags_ecg_1 = tags_ecg_feas1
        self.tags_pt_1 = tags_pt_feas1
        self.tags_ecg_2 = tags_ecg_feas2
        self.tags_pt_2 = tags_pt_feas2
        self.tags_ecg_t = tags_ecg_trial
        self.tags_pt_t = tags_pt_trial

        self.tags_pt_ecg_1 = np.array(pd.DataFrame(self.tags_ecg_1, columns=['ptID', 'measNo', 'measID']))
        self.tags_pt_ecg_2 = np.array(pd.DataFrame(self.tags_ecg_2, columns=['ptID', 'measNo', 'measID']))
        self.tags_pt_ecg_t = np.array(pd.DataFrame(self.tags_ecg_t, columns=['ptID', 'measNo', 'measID']))
        self.tags_pt_info_1 = np.array(pd.DataFrame(self.tags_pt_1, columns=['ptID', 'ptDiag']))
        self.tags_pt_info_2 = np.array(pd.DataFrame(self.tags_pt_2, columns=['ptID', 'ptDiag']))
        self.tags_pt_info_t = np.array(pd.DataFrame(self.tags_pt_t, columns=['ptID', 'ptDiag']))
        self.tags_ecg_info_1 = np.array(pd.DataFrame(self.tags_ecg_1, columns=['measID', 'measDiag', 'ptDiag', 'measDiagAgree']))
        self.tags_ecg_info_2 = np.array(pd.DataFrame(self.tags_ecg_2, columns=['measID', 'measDiag', 'ptDiag']))
        self.tags_ecg_info_t = np.array(pd.DataFrame(self.tags_ecg_t, columns=['measID', 'measDiag', 'ptDiag', 'measDiagAgree']))

        self.key_1 = key_1
        self.key_2 = key_2
        self.key_t = key_t
        self.ecg_num_1 = 162515
        self.ecg_num_2 = 23253
        self.ecg_num_t = 1104005
        self.pt_num_1 = 2141
        self.pt_num_2 = 288
        self.pt_num_t = 13453

    def extract_signal(self, measID_num, study = 'feas2'):
        if study == 'feas2':
            if measID_num > self.ecg_num_2: raise Exception('Not a valid measID')
            measID_vis = '0'*(6-len(str(measID_num)))+str(measID_num)
            folder_vis = '0'*(3-len(str(measID_num//1000)))+str(measID_num//1000)+'0'*3
            ecg_signal = wfdb.rdrecord('Data\\ECGs\\feas2\\'+folder_vis+'\\saferF2_'+measID_vis)
            return np.ndarray.flatten(ecg_signal.p_signal)
        elif study == 'feas1':
            if measID_num > self.ecg_num_1: raise Exception('Not a valid measID')
            # for i in self.tags_pt_ecg_1:
            #     if i[-1] == measID_num:
            #         ptID_num = i[0]
            #         measNo_num = i[1]
            # ptID_vis = '0'*(6-len(str(ptID_num)))+str(ptID_num)
            # folder_vis = '0'*(4-len(str(ptID_num//100)))+str(ptID_num//100)+'0'*2
            # ecg_signal = wfdb.rdrecord('Data\\ECGs\\feas1\\'+folder_vis+'\\saferF1_pt'+ptID_vis)
            measID_vis = '0'*(6-len(str(measID_num)))+str(measID_num)
            ecg_signal = wfdb.rdrecord('Data\\ECGs\\feas1\\afECGs\\saferF1_'+measID_vis)
            return np.ndarray.flatten(ecg_signal.p_signal)
            # return np.ndarray(ecg_signal)[:,measNo_num-1]

    def extract_tags(self, measID_num, study = 'feas2'):
        if study == 'feas2':
            if measID_num > self.ecg_num_2: raise Exception('Not a valid measID')
            tags = self.tags_ecg_2.iloc[measID_num-1]
        elif study == 'feas1':
            if measID_num > self.ecg_num_1: raise Exception('Not a valid measID')
            tags = self.tags_ecg_1.iloc[measID_num-1]
        assert tags['measID'] == measID_num                   
        return tags
    
    def extract_label(self, tags):
        return tags['measDiag']
    
    def extract_cardiolund_tags(self, tags, study = 'feas2'):
        if study == 'feas2':
            cld_tags = list(tags.iloc[22:37]) 
        elif study == 'feas1':
            cld_tags =  list(tags.iloc[25:40])
        if 1 in [1 for i in cld_tags if math.isnan(i)]:
            return None
        else:
            return cld_tags
    
    def extract_pt_measIDs(self, ptID_num, study = 'feas2'):
        if study == 'feas2':
            tags_ecg = self.tags_pt_ecg_2
            ecg_num = self.ecg_num_2
        elif study == 'feas1':
            tags_ecg = self.tags_pt_ecg_1
            ecg_num = self.ecg_num_1
        ptID_measIDs = []
        i = [j[0] for j in tags_ecg].index(ptID_num)
        while i < ecg_num and tags_ecg[i][0] == ptID_num:
            ptID_measIDs.append(tags_ecg[i][-1])
            i += 1
        return ptID_measIDs
    
    def extract_measID_pt(self, measID_num, study = 'feas2'):
        if study == 'feas2':
            tags_ecg = self.tags_pt_ecg_2
        elif study == 'feas1':
            tags_ecg = self.tags_pt_ecg_1
        return [j[0] for j in tags_ecg if j[2] == measID_num][0]
    
    def extract_measIDs(self, tag = 'AF', study = 'feas2'):
        if study == 'feas2':
            tags_ecg = self.tags_ecg_info_2
            key = self.key_2
        elif study == 'feas1':
            tags_ecg = self.tags_ecg_info_1
            key = self.key_1
        diag = key[tag]
        if tag == 'undecided':
            diag_noAF = key['noAF']
            return [i[0] for i in tags_ecg if (i[1] == diag and i[2] == diag_noAF)]
        else:
            return [i[0] for i in tags_ecg if i[1] == diag]
    
    def extract_ptIDs(self, tag = 'AF', study = 'feas2'):
        if study == 'feas2':
            tags_ecg = self.tags_pt_info_2
            key = self.key_2
        elif study == 'feas1':
            tags_ecg = self.tags_pt_info_1
            key = self.key_1
        diag = key[tag]
        return [i[0] for i in tags_ecg if i[1] == diag]
    
    def store_ecg_tags(self, tag = 'AF', study = 'feas2', measIDs = None):
        if measIDs is None: 
            measID_AF = self.extract_measIDs(tag=tag, study=study)
        else:
            measID_AF = measIDs
        data = []
        for i in measID_AF:
            tags = self.extract_tags(i, study = study)
            cdl_tags = self.extract_cardiolund_tags(tags, study = study)
            if cdl_tags is not None:
                data.append(cdl_tags)
        return data


def generate_filter(fs=500, low_cut=0.6, high_cut=40, N=10):
    return signal.butter(N=N, Wn=[low_cut, high_cut], btype='bandpass', analog=False, output='sos', fs=fs)

def apply_filters(ecg_signal, sos):
    return signal.sosfilt(sos, ecg_signal)

def filter_signal(ecg_signal, sos):
    n = 0
    if len(ecg_signal) != 15200: n = 15200 - len(ecg_signal)
    ecg_signal = np.concatenate((ecg_signal, np.zeros(n)))
    DWTcoeffs = pywt.wavedec(ecg_signal, 'sym6')
    for i in range(1,8):
        DWTcoeffs[-i] = np.zeros_like(DWTcoeffs[-i])
    ecg_filtered = np.array(ecg_signal) - pywt.waverec(DWTcoeffs,'sym6',mode='symmetric',axis=-1)
    for _ in range(5):
        ecg_filtered = apply_filters(ecg_filtered, sos)
    return ecg_filtered

def normalise_signals(ecg_signals):
    norm_ecg_signals = []
    for i in ecg_signals:
        norm_ecg_signals.append((np.array(i)-np.mean(i))/np.std(i))
    return norm_ecg_signals
