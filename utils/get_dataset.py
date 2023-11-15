import pandas as pd
import numpy as np

def get_fb_data(n_pairs):
    pairdata = {}
    freqpairs = []
    cdf_size = pd.read_csv('dataset/fb/size_cdf.csv')
    cdf_interval = pd.read_csv('dataset/fb/interval_cdf.csv')
    n_size, n_interval = len(cdf_size), len(cdf_interval)
    for i in range(n_pairs):
        data = pd.read_csv('dataset/fb/data/' + 'ip_pair_{i}.csv'.format(i=i))
        pair = data['ip_pair'].iloc[0]
        pairdata[pair] = data
        freqpairs.append(pair)
    return pairdata, freqpairs, n_size, n_interval


def get_univ_data(n_pairs):
    pairdata = {}
    data = pd.read_csv('dataset/univ/data.csv')
    cdf_size = pd.read_csv('dataset/univ/size_cdf.csv')
    cdf_interval = pd.read_csv('dataset/univ/interval_cdf.csv')
    n_size, n_interval = len(cdf_size), len(cdf_interval)
    pairs, counts = np.unique(data['src_dst'], return_counts=True)
    freqpairs = list(pairs[np.argsort(counts)[-n_pairs:]])
    for pair in freqpairs:
        pairdata[pair] = data[data['src_dst'] == pair]
    return pairdata, freqpairs, n_size, n_interval


def get_data(pairdata, freqpairs, index, n):
    pairs = len(pairdata)
    data = np.zeros((pairs, n)) 
    for pair in range(pairs):
        feature = pairdata[freqpairs[pair]][index].iloc[:-1]
        freq = feature.value_counts().sort_index()
        for key, value in freq.items():
            data[pair][key] = value
    data /= data.sum(axis=1).reshape(-1, 1)
    return data


def get_trans(pairdata, freqpairs, index, n):
    pairs = len(pairdata)
    s2s_pair = np.zeros((pairs, n * n))
    for pair in range(pairs):
        sizeindex = pairdata[freqpairs[pair]][index].values
        feature = sizeindex[0:-1] * n + sizeindex[1:]
        values, counts = np.unique(feature, return_counts=True)
        s2s_pair[pair][values] = counts
    s2s_pair /= s2s_pair.sum(axis=1).reshape(-1, 1)

    size_trans = np.zeros((pairs, n, n))
    for pair in range(pairs):
        s2s = np.zeros((n, n))
        sizeindex = pairdata[freqpairs[pair]][index].values
        for i in range(len(sizeindex) - 1):
            s2s[sizeindex[i]][sizeindex[i + 1]] += 1
        s2s[sizeindex[-1]][sizeindex[0]] += 1
        s2s = np.divide(s2s, s2s.sum(axis=1).reshape(-1, 1), out=np.zeros_like(s2s), where=s2s.sum(axis=1).reshape(-1, 1) != 0)
        size_trans[pair] = s2s
    return s2s_pair, size_trans