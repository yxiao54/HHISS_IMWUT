import copy
import random
import numpy as np
import os
import pandas as pd
from functools import reduce
from sklearn.preprocessing import StandardScaler,Normalizer,normalize,MinMaxScaler,QuantileTransformer
from torch.utils.data import Sampler
class RandomDomainSampler(Sampler):
    r"""Randomly sample :math:`N` domains, then randomly select :math:`K` samples in each domain to form a mini-batch of
    size :math:`N\times K`.

    Args:
        data_source (ConcatDataset): dataset that contains data from multiple domains
        batch_size (int): mini-batch size (:math:`N\times K` here)
        n_domains_per_batch (int): number of domains to select in a single mini-batch (:math:`N` here)
    """

    def __init__(self, data_source, batch_size, n_domains_per_batch):
        super(Sampler, self).__init__()
        
        
        unique_users=np.unique(data_source)
        
        self.n_domains_in_dataset = len(unique_users)
        self.n_domains_per_batch = n_domains_per_batch
        assert self.n_domains_in_dataset >= self.n_domains_per_batch
        self.sample_idxes_per_domain=[]
        for user in unique_users:
            user_idx=np.argwhere(data_source==user) 
            user_idx=user_idx.reshape(-1).tolist()
            self.sample_idxes_per_domain.append(user_idx)
        


        assert batch_size % n_domains_per_batch == 0
        self.batch_size_per_domain = batch_size // n_domains_per_batch
        self.length = len(list(self.__iter__()))

    def __iter__(self):
        sample_idxes_per_domain = copy.deepcopy(self.sample_idxes_per_domain)
        domain_idxes = [idx for idx in range(self.n_domains_in_dataset)]
        final_idxes = []
        stop_flag = False
        while not stop_flag:
            selected_domains = random.sample(domain_idxes, self.n_domains_per_batch)

            for domain in selected_domains:
                sample_idxes = sample_idxes_per_domain[domain]
         
                if len(sample_idxes) < self.batch_size_per_domain:
                    selected_idxes = np.random.choice(sample_idxes, self.batch_size_per_domain, replace=True)
                else:
                    selected_idxes = random.sample(sample_idxes, self.batch_size_per_domain)
                final_idxes.extend(selected_idxes)

                for idx in selected_idxes:
                    if idx in sample_idxes_per_domain[domain]:
                        sample_idxes_per_domain[domain].remove(idx)

                remaining_size = len(sample_idxes_per_domain[domain])
                if remaining_size < self.batch_size_per_domain:
                    stop_flag = True

        return iter(final_idxes)

    def __len__(self):
        return self.length
        

def user_changeScore(select_data,select_user,select_label,select_task):

    b,c,d=None,None,None
    if len(select_data.shape)==3:
        b,c,d=select_data.shape
        select_data=select_data.reshape(b*c,d)
        
    select_label=np.asarray(select_label)
    unique_users=np.unique(select_user)
    
    all_data=[]
    all_user=[]
    all_label=[]
    all_task=[]
    for user in unique_users:
        user_idx=np.argwhere(select_user==user)
        label_idx=np.argwhere(select_label==0)
        
        
        baseline_idx=reduce(np.intersect1d, [user_idx, label_idx])
        
        baseline_idx=baseline_idx[:3]
         
        user_idx=np.setdiff1d(user_idx,baseline_idx)
        
        
        user_data=select_data[user_idx]
        baseline_data=select_data[baseline_idx]
        
        scaler = MinMaxScaler().fit(baseline_data)
        user_data = scaler.transform(user_data)
        
        user_user=select_user[user_idx]
        user_label=select_label[user_idx]
        user_task=select_task[user_idx]
        
        all_data.append(user_data)
        all_user.append(user_user)
        all_label.append(user_label)
        all_task.append(user_task)
            
    all_data=np.concatenate(all_data,0)
    all_user=np.concatenate(all_user,0)
    all_label=np.concatenate(all_label,0)
    all_task=np.concatenate(all_task,0)
    

    if b!=None:
        select_data=select_data.reshape(b,c,d)
        
    return all_data,all_user,all_label,all_task


def user_changeScore2(select_data,select_user,select_label,select_task):

    b,c,d=None,None,None
    if len(select_data.shape)==3:
        b,c,d=select_data.shape
        select_data=select_data.reshape(b*c,d)
        
    select_label=np.asarray(select_label)
    unique_users=np.unique(select_user)
    
    all_data=[]
    all_user=[]
    all_label=[]
    all_task=[]
    for user in unique_users:
        user_idx=np.argwhere(select_user==user)
        label_idx=np.argwhere(select_label==0)
        
        
        baseline_idx=reduce(np.intersect1d, [user_idx, label_idx])
        
        baseline_idx=baseline_idx[:3]
         
        user_idx=np.setdiff1d(user_idx,baseline_idx)
        
        
        user_data=select_data[user_idx]
        baseline_data=select_data[baseline_idx]
        
        scaler = StandardScaler().fit(baseline_data)
        user_data = scaler.transform(user_data)
        
        user_user=select_user[user_idx]
        user_label=select_label[user_idx]
        user_task=select_task[user_idx]
        
        all_data.append(user_data)
        all_user.append(user_user)
        all_label.append(user_label)
        all_task.append(user_task)
            
    all_data=np.concatenate(all_data,0)
    all_user=np.concatenate(all_user,0)
    all_label=np.concatenate(all_label,0)
    all_task=np.concatenate(all_task,0)
    

    if b!=None:
        select_data=select_data.reshape(b,c,d)
        
    return all_data,all_user,all_label,all_task
    

def user_standar(select_data,select_user,select_label):

    b,c,d=None,None,None
    if len(select_data.shape)==3:
        b,c,d=select_data.shape
        select_data=select_data.reshape(b*c,d)
        
        
    unique_users=np.unique(select_user)
    for user in unique_users:
        user_idx=np.argwhere(select_user==user)
        user_data=select_data[user_idx]
        user_data=np.squeeze(user_data)
        
        scaler = StandardScaler().fit(user_data)
        user_data = scaler.transform(user_data)
       
 
        user_data=np.expand_dims(user_data,1)

        select_data[user_idx]=user_data
        
    if b!=None:
        select_data=select_data.reshape(b,c,d)
        
    return select_data
def user_minmax(select_data,select_user,select_label):
    b,c,d=None,None,None
    if len(select_data.shape)==3:
        b,c,d=select_data.shape
        select_data=select_data.reshape(b*c,d)
        
    unique_users=np.unique(select_user)
    for user in unique_users:
        user_idx=np.argwhere(select_user==user)
        user_data=select_data[user_idx]
        user_data=np.squeeze(user_data)
        
        scaler = MinMaxScaler().fit(user_data)
        user_data = scaler.transform(user_data)
       
 
        user_data=np.expand_dims(user_data,1)

        select_data[user_idx]=user_data
        
    if b!=None:
        select_data=select_data .reshape(b,c,d)
    return select_data
    

    

def minmax_norm(train_datas,val_data,test_data):
    b,c,d=None,None,None
    if len(train_datas.shape)==3:
        b,c,d=train_datas.shape
        train_datas=train_datas.reshape(b*c,d)
        
        bt,ct,dt=test_data.shape
        test_data=test_data.reshape(bt*ct,dt)
        
        bv,cv,dv=val_data.shape
        val_data=val_data.reshape(bv*cv,dv)
        
    scaler = MinMaxScaler().fit(train_datas)
    train_datas = scaler.transform(train_datas)
    test_data = scaler.transform(test_data)
    if b!=None:
        train_datas=train_datas.reshape(b,c,d)
        test_data=test_data.reshape(bt,ct,dt)
        val_data=val_data.reshape(bv,cv,dv)
        
    
    return train_datas,val_data,test_data

def standard_norm(train_datas,val_data,test_data):
    b,c,d=None,None,None
    if len(train_datas.shape)==3:
        b,c,d=train_datas.shape
        train_datas=train_datas.reshape(b*c,d)
        
        bt,ct,dt=test_data.shape
        test_data=test_data.reshape(bt*ct,dt)
        
        bv,cv,dv=val_data.shape
        val_data=val_data.reshape(bv*cv,dv)
        
    scaler = StandardScaler().fit(train_datas)
    train_datas = scaler.transform(train_datas)
    test_data = scaler.transform(test_data)
    
    if b!=None:
        train_datas=train_datas.reshape(b,c,d)
        test_data=test_data.reshape(bt,ct,dt)
        val_data=val_data.reshape(bv,cv,dv)
    
    return train_datas,val_data,test_data