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
from torchsummary import summary
from utils.get_dataset import *
from contextlib import nullcontext
from dataclasses import dataclass
import datetime  
device = 'cpu'

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
        z = reparameterize(mu, var)
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


def get_locality(pairdata, freqpairs):
    src_count, dst_count = defaultdict(int), defaultdict(int)
    src_ip, dst_ip = [], []

    for pair in freqpairs:
        src_ip.append(pairdata[pair]['srcip'][0])
        dst_ip.append(pairdata[pair]['dstip'][0])
        src_count[pairdata[pair]['srcip'][0]] += 1
        dst_count[pairdata[pair]['dstip'][0]] += 1

    max_src_count = np.max(list(src_count.values()))
    max_dst_count = np.max(list(dst_count.values()))
    condition_size = max_src_count + max_dst_count + 2

    locality_strings = []
    locality_onehots = []
    for pair in freqpairs:
        locality_strings.append("{src},{dst}".format(src=src_count[pairdata[pair]['srcip'][0]], dst=dst_count[pairdata[pair]['dstip'][0]]))
        locality_onehot = np.zeros(condition_size)
        locality_onehot[src_count[pairdata[pair]['srcip'][0]]] = 1
        locality_onehot[max_src_count + dst_count[pairdata[pair]['dstip'][0]]] = 1
        locality_onehots.append(locality_onehot)
    
    values, counts = np.unique(locality_strings, return_counts=True)
    pair_counts = dict(zip(values, counts))

    return locality_strings, np.array(locality_onehots), pair_counts, condition_size


if __name__ == "__main__":
    t0 = time.time()
    # read data
    traces = 100000
    pairdata, freqpairs, n_size, n_interval = get_fb_data(traces)
    sizedata = get_data(pairdata, freqpairs, 'size_index', n_size)
    intervaldata = get_data(pairdata, freqpairs, 'interval_index', n_interval)
    print('read data in %dm %ds' % ((time.time() - t0) / 60, (time.time() - t0) % 60))

    # get locality
    locality_strings, locality_onehots, pair_counts, condition_size = get_locality(pairdata, freqpairs)
    print('get locality in %dm %ds' % ((time.time() - t0) / 60, (time.time() - t0) % 60))
    dataset = [pair for pair in zip(sizedata, locality_onehots)]

    hidden_dims = [768, 512, 256]
    latent_dim = 32
    encoder = SizeEncoder(n_size, condition_size, hidden_dims, latent_dim).to(device)
    hidden_dims.reverse()
    decoder = SizeDecoder(latent_dim, condition_size, hidden_dims, n_size).to(device)
    print('encoder:', summary(encoder, [[n_size], [condition_size]], device=device))
    print('decoder:', summary(decoder, [[latent_dim], [condition_size]], device=device))
    sys.stdout.flush()

    lr = 1e-4
    kld_weight = 1e-5
    optimizer = torch.optim.Adam([{'params': encoder.parameters()}, {'params': decoder.parameters()}], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
    print('start in:', time.strftime("%Y/%m/%d %H:%M:%S", time.localtime()))
    sys.stdout.flush()

    date = datetime.datetime.now()
    date = '%s-%s-%s-%s' % (date.year, date.month, date.day, date.hour)
    if os.path.exists('model/{date}/'.format(date=date)):
        os.system('rm -r model/{date}/'.format(date=date))
    os.makedirs('model/{date}/'.format(date=date))

    start_time = time.time()
    avg_loss = 0
    min_loss = 1e9
    print_every = 100
    for epoch in range(100001):
        epoch_loss, epoch_recon, epoch_kld, max_loss = train(encoder, decoder, dataset, optimizer)
        avg_loss += epoch_loss
        if epoch and epoch % print_every == 0:
            avg_loss /= print_every
            cur_time = time.time()
            print("epoch=%d, avg_loss=%.2e, kld=%.2f, recon=%.2e(max=%.2e), time=%.2f" % (epoch, avg_loss, epoch_kld, epoch_recon, max_loss, cur_time - start_time))
            if avg_loss < min_loss:
                min_loss = avg_loss
                torch.save(encoder, 'model/{date}/encoder.pth'.format(date=date))
                torch.save(decoder, 'model/{date}/decoder.pth'.format(date=date))
                print('save model')
            sys.stdout.flush()
            if avg_loss < 1e-3:
                break
            avg_loss = 0

    