import os
import sys
import pandas as pd
import numpy as np
import time         
from sklearn import metrics
import math
from collections import defaultdict
import torch
import torch.nn as nn
from torch import nn, Tensor
from typing import List
from torch.nn import functional as F
from torch.utils.data import DataLoader
# from torchsummary import summary
from utils.get_dataset import *
from contextlib import nullcontext
from dataclasses import dataclass
import datetime
import random  
import matplotlib.pyplot as plt
device = 'cuda'

class SizeEncoder(nn.Module):
    def __init__(self, input_dim, condition_dim, hidden_dims, latent_dim):
        super(SizeEncoder, self).__init__()
        self.encoder = nn.ModuleList()
        in_dim = input_dim + condition_dim
        for h_dim in hidden_dims:
            self.encoder.append(
                nn.Sequential(
                    nn.Linear(in_dim, out_features=h_dim),
                    nn.ReLU())
                    # ,nn.LayerNorm(h_dim))
            )
            in_dim = h_dim
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
    
    def forward(self, x: Tensor, c: Tensor) -> List[Tensor]:
        x = torch.cat((x, c), dim=1)
        for module in self.encoder:
            x = module(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return [mu, log_var]


class SizeDecoder(torch.nn.Module):
    def __init__(self, latent_dim, condition_dim, hidden_dims, output_dim):
        super(SizeDecoder, self).__init__()
        self.decoder = torch.nn.ModuleList()
        in_dim = latent_dim + condition_dim
        for h_dim in hidden_dims:
            self.decoder.append(
                nn.Sequential(
                    nn.Linear(in_dim, out_features=h_dim,),
                    nn.ReLU())
                    # nn.LayerNorm(h_dim))
            )
            in_dim = h_dim
        self.output = nn.Linear(hidden_dims[-1], output_dim)
    
    def forward(self, x: Tensor, c: Tensor) -> List[Tensor]:
        x = torch.cat((x, c), dim=1)
        for module in self.decoder:
            x = module(x)
        result = self.output(x)
        result = F.softmax(result, dim=1)
        return result

def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu


def train(encoder, decoder, dataset, optimizer):
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    encoder.train()
    decoder.train()
    epoch_loss, epoch_kld, epoch_recon, sample_num, epoch_max_loss = 0, 0, 0, 0, 0
    max_loss_weight = 0.1
    for size_data, condition in dataloader:
        optimizer.zero_grad()
        size_data, condition = size_data.float().to(device), condition.float().to(device)
        mu, var = encoder(size_data, condition)
        z = reparameterize(mu, var)#z.shape=(batch_size,latent_dim)
        y = decoder(z, condition)
        recon_loss = F.l1_loss(y, size_data)
        max_recon_loss = torch.max(torch.abs(y - size_data).mean(dim=1))
        kld_loss = torch.mean(-0.5 * torch.sum(1 + var - mu ** 2 - var.exp(), dim = 1), dim = 0)
        loss = recon_loss + max_loss_weight * max_recon_loss + kld_weight * kld_loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(size_data)
        epoch_kld += kld_loss.item() * len(size_data)
        epoch_recon += recon_loss.item() * len(size_data)
        epoch_max_loss = max(epoch_max_loss, max_recon_loss.item())
        sample_num += len(size_data)

    epoch_loss /= sample_num
    epoch_recon /= sample_num
    epoch_kld /= sample_num
    return epoch_loss, epoch_recon, epoch_kld, epoch_max_loss


def evaluate(decoder,sizedata,locality,latent_dim,locality_onehots_dict,step=0, test_size=100):
    t0=time.time()
    # bins=20
    decoder.eval()
    with torch.no_grad():
        condition = random.sample(locality_onehots_dict.keys(),min(len(locality_onehots_dict.keys()),500))
        condition = [eval(i) for i in condition]
        condition=torch.Tensor(np.array(condition)).float().to(device)
        c_min_99 = []
        c_min_aver = []
        r_min_99 = []
        r_min_aver = []
        coverage = []
        for cond in condition:
            z=torch.randn([test_size,latent_dim]).to(device)
            y=decoder(z,cond.expand(test_size,-1)).cpu()
        # print(y.shape)#y.shape=[test_size, 类别数=65]
            locality_key = str(list(np.array(cond.cpu())))
            matrix=torch.zeros((test_size,len(locality_onehots_dict[locality_key])))
            for i in range(test_size):
                for j in range(len(locality_onehots_dict[locality_key])): 
                    matrix[i,j]=torch.max(np.abs(y[i]-sizedata[locality_onehots_dict[locality_key][j]]))   
            # for j in range(len(dataset)):
            #     matrix[i][j] = sum(abs(y[i]-dataset[j][0]))/dataset[j][0].shape[0]
            column_min,column_posi=torch.min(matrix,dim=0)
            row_min,row_posi=torch.min(matrix,dim=1)
            c_min_99.append(np.percentile(column_min,99))
            c_min_aver.append(np.mean(np.array(column_min)))
            r_min_99.append(np.percentile(row_min,99))
            r_min_aver.append(np.mean(np.array(row_min)))
            row_posi=torch.unique(row_posi)
            coverage.append(len(row_posi)/len(locality_onehots_dict[locality_key]))
        # plt.hist(column_min,bins=bins,density=True,cumulative=True,color='blue')
        # plt.hist(row_min,bins=bins,density=True,cumulative=True,color='yellow')
        # plt.savefig('result/{date}/'.format(date=date)+str(step)+".png")

        # print("coverage for testsize "+str(test_size)+" is :"+str(len(row_posi)/test_size))
        print("eval in"+str(time.time()-t0)+" //coverage is %.4f on average and is %.4f for the worst" %
              (np.mean(coverage), np.percentile(coverage,1)) +
              " //per sample error is %.4f on average and is %.4f for the worst99 and is %.4f for the worst50" %(np.mean(r_min_aver),np.percentile(r_min_99,99),np.percentile(r_min_99,50)) +
              " //per source data error is %.4f on average and is %.4f for the worst99 and is %.4f for the worst50" %(np.mean(c_min_aver),np.percentile(c_min_99,99),np.percentile(c_min_99,50)))
        # plt.close()




def get_locality(pairdata, freqpairs,pairsize):
    latent_size = 128
    
    src_count, dst_count = defaultdict(int), defaultdict(int)
    src_ip, dst_ip = [], []

    for pair in freqpairs:
        src_ip.append(pairdata[pair]['srcip'][0])
        dst_ip.append(pairdata[pair]['dstip'][0])
        src_count[pairdata[pair]['srcip'][0]] += 1
        dst_count[pairdata[pair]['dstip'][0]] += 1

    max_src_count = np.max(list(src_count.values()))
    max_dst_count = np.max(list(dst_count.values()))
    print("maxsrccount"+str(max_src_count)+"maxdstcount"+str(max_dst_count))
    condition_size = max_src_count + max_dst_count + 2

    locality_strings = []
    locality_onehots = []
    locality_onehots_dict=defaultdict(list)
    i = 0
    for pair in freqpairs:
        locality_strings.append("{src},{dst}".format(src=src_count[pairdata[pair]['srcip'][0]], dst=dst_count[pairdata[pair]['dstip'][0]]))
        locality_onehot = np.zeros(condition_size)
        locality_onehot[src_count[pairdata[pair]['srcip'][0]]] = 1
        locality_onehot[max_src_count + dst_count[pairdata[pair]['dstip'][0]]] = 1
        locality_onehots.append(locality_onehot)
        # locality_onehot.dtype=int
        locality_onehots_dict[str(list(locality_onehot))].append(i)
        i+=1
    values, counts = np.unique(locality_strings, return_counts=True)
    pair_counts = dict(zip(values, counts))

    return locality_strings, np.array(locality_onehots), pair_counts, condition_size, locality_onehots_dict

def get_kld_weight(epoch=0):
    kld_max=1e-4
    kld_min=1e-6
    if epoch and epoch%2000==0:
        return kld_max
    epoch%=2000
    if epoch<1000:
        return (kld_max-kld_min)*((epoch)/1000.0)+kld_min
    else:
        return kld_max

if __name__ == "__main__":
    t0 = time.time()
    random.seed(114514)
    # read data
    traces = 100000
    pairdata, freqpairs, n_size, n_interval,pairsize = get_fb_data(traces)
    sizedata = get_data(pairdata, freqpairs, 'size_index', n_size)
    intervaldata = get_data(pairdata, freqpairs, 'interval_index', n_interval)
    print('read data in %dm %ds' % ((time.time() - t0) / 60, (time.time() - t0) % 60))

    # get locality
    locality_strings, locality_onehots, pair_counts, condition_size, locality_onehots_dict = get_locality(pairdata, freqpairs,pairsize)
    print('get locality in %dm %ds' % ((time.time() - t0) / 60, (time.time() - t0) % 60))
    dataset = [pair for pair in zip(sizedata, locality_onehots)]
    # print(dataset[0][0].shape)
    # exit()
    
    hidden_dims = [768, 512, 256]
    latent_dim = 32
    encoder = SizeEncoder(n_size, condition_size, hidden_dims, latent_dim).to(device)
    hidden_dims.reverse()
    decoder = SizeDecoder(latent_dim, condition_size, hidden_dims, n_size).to(device)
    # print('encoder:', summary(encoder, [[n_size], [condition_size]], device=device))
    # print('decoder:', summary(decoder, [[latent_dim], [condition_size]], device=device))
    sys.stdout.flush()

    lr = 1e-4
    kld_weight = 0#1e-5#1e-4不行
    optimizer = torch.optim.Adam([{'params': encoder.parameters()}, {'params': decoder.parameters()}], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
    print('start in:', time.strftime("%Y/%m/%d %H:%M:%S", time.localtime()))
    sys.stdout.flush()

    date = datetime.datetime.now()
    date = '%s-%s-%s-%s' % (date.year, date.month, date.day, date.hour)
    if os.path.exists('model/{date}/'.format(date=date)):
        os.system('rm -r model/{date}/'.format(date=date))
    os.makedirs('model/{date}/'.format(date=date))
    if os.path.exists('result/{date}/'.format(date=date)):
        os.system('rm -r result/{date}/'.format(date=date))
    os.makedirs('result/{date}/'.format(date=date))

    start_time = time.time()
    avg_loss = 0
    min_loss = 1e9
    print_every = 100
    for epoch in range(100001):
        kld_weight = get_kld_weight(epoch)
        epoch_loss, epoch_recon, epoch_kld, max_loss = train(encoder, decoder, dataset, optimizer)
        avg_loss += epoch_loss
        if epoch and epoch % print_every == 0:
            avg_loss /= print_every
            cur_time = time.time()
            print("epoch=%d, avg_loss=%.2e, kld=%.2f, recon=%.2e(max=%.2e), time=%.2f" % (epoch, avg_loss, epoch_kld, epoch_recon, max_loss, cur_time - start_time))
            if epoch % (print_every*10) == 0:
                evaluate(decoder,sizedata,locality_onehots,latent_dim,locality_onehots_dict,epoch,1000)
            else:
                evaluate(decoder,sizedata,locality_onehots,latent_dim,locality_onehots_dict,epoch,100)
            if avg_loss < min_loss:
                min_loss = avg_loss
                torch.save(encoder, 'model/{date}/encoder.pth'.format(date=date))
                torch.save(decoder, 'model/{date}/decoder.pth'.format(date=date))
                print('save model')
            sys.stdout.flush()
            if avg_loss < 1e-3:
                break
            avg_loss = 0

    