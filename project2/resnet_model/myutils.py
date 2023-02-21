import numpy as np
import pandas as pd

import neurokit2 as nk

import os
from multiprocessing import Pool
from tqdm import tqdm

import json

def fix_length_and_clean(signal):
    signal, is_inverted = nk.ecg_invert(signal, sampling_rate=300, show=False)
    c_signal = nk.ecg_clean(signal, sampling_rate=300, method='neurokit')
    fixed_size = nk.signal_resample(c_signal, desired_length=6000, sampling_rate=300, desired_sampling_rate=300, method='FFT')
    return fixed_size

def get_features(df_raw_signals):
    
    features = []
    
    for i in tqdm(range(0, df_raw_signals.shape[0])):
        signal = df_raw_signals.iloc[i].dropna().to_numpy(dtype='float32')
        f = fix_length_and_clean(signal)
        features.append(f)
    
    df = pd.DataFrame(features)
    return df

def sub_features(arg_tuple):
    df_raw, idx = arg_tuple
    df_processed = get_features(df_raw)
    return idx, df_processed

def multi_features(df_raw_signals, n_cores=128):
    ids = df_raw_signals.index.to_list()
    split = np.array_split(ids, n_cores)
    
    chunks = []
    for l, i in zip(split, range(len(split))):
        start = l[0]
        end = l[-1]
        chunks.append((df_raw_signals.iloc[start:end+1], i))
    
    my_pool = Pool(n_cores)
    result = my_pool.map(sub_features, chunks)
    result = sorted(result, key=lambda tup: tup[0])
    
    df_list = [item[1] for item in result]
    df_final = pd.concat(df_list)
    df_final = df_final.reset_index(drop=True)
    
    return df_final