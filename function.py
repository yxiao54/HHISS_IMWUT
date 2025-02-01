import torch
from torch import nn
import numpy as np
import os
import pandas as pd
from functools import reduce
import pickle
import myutils as utils



class H5Dataset(torch.utils.data.Dataset):

    def __init__(self, data,label,task,user,dataset_name):
        self.data=data
        self.label=label
        self.task=task
        self.user=user
        self.dataset_name=dataset_name
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
 
        return self.data[index],self.label[index],self.task[index],self.user[index]
        
def read_pickle(modality_name):

    path='/home/yxiao124/mobicom/data/all_data.csv'

    data_df = pd.read_csv(path) 
    
    
    
    
    all_quality=data_df.loc[:,"ppg_quality"].to_numpy()

    all_label=data_df.loc[:,"label"].to_numpy()
    all_user=data_df.loc[:,"user"].to_numpy()
    all_task=data_df.loc[:,"task"].to_numpy()
    
    all_data=data_df.loc[:, ~data_df.columns.isin(['ppg_quality', 'label',"user","task",'Unnamed: 0'])]#.to_numpy()
    
    
    #m1 = np.quantile(all_quality, 0.95, axis=0)
    #m2 = np.quantile(all_quality, 0.05, axis=0)
    
    #plt.hist(all_quality)
    #plt.savefig('./quality.png')
    #plt.cla()
    #plt.clf()
    
    
    
    drop_list=['HRV_SDANN1','HRV_SDNNI1','HRV_SDANN2','HRV_SDNNI2','HRV_SDANN5','HRV_SDNNI5','HRV_VLF','HRV_ULF','Phasic_phasic_entropy','BVP_BVP_entropy']
    all_data=all_data.drop(drop_list, axis=1)
    
    all_data.replace([np.inf, -np.inf], np.nan, inplace=True)
     
    all_data=all_data.fillna(0)
   
    #nan_count = all_data.isin([np.inf, -np.inf]).sum()
    
    #nan_count.to_csv('nan_count.csv') 
    #"all",'eda','edabvp','edahr','edahrv','edatemp','edaacc'
    if modality_name=='all':
        
        modality=['EDA','HRV','Phasic','Tonic','TEMP','HR','ACC','BVP']
    elif modality_name=='eda':
        modality=['EDA','Phasic','Tonic']
    elif modality_name=='edabvp':
        modality=['EDA','Phasic','Tonic','BVP']
    elif modality_name=='edahr':
        modality=['EDA','Phasic','Tonic','HR']
    elif modality_name=='edahrv':
        modality=['EDA','Phasic','Tonic','HRV']
    elif modality_name=='edatemp':
        modality=['EDA','Phasic','Tonic','TEMP']
    elif modality_name=='edaacc':
        modality=['EDA','Phasic','Tonic','ACC']
  
    
    names=all_data.columns
    keep=[]

    for name in names:
        moda=name.split("_")[0]
        if moda in modality:
            keep.append(name)
    
    #with open("result1.pickle", "rb") as output_file:
    #    keep=pickle.load(output_file)
    #all_data=all_data.loc[:,all_data.columns.isin(keep)]#.to_numpy()
    all_data=all_data.loc[:,all_data.columns.isin(keep)]


            
    all_data=all_data.to_numpy()
    #flag1=np.isnan(all_data).any()
    #flag2=np.isinf(all_data).any()
    
    



    
    return all_data,all_label,all_user,all_task,all_quality
       
    
    

    
def get_dataloader(norm_name,modality_name,irm_batch=False,batch_size=50):
    
    label_dict={'calm':0,'stress':1}
    task_set=['jelly','count','stress','prepareSong','arithmetic','bad','calm','cycling1','cycling2','run1','run2','sit','stand','baseline','stroop','HAHV_sit','LAHV_sit','LALV_sit','HALV_sit','HAHV_walk','LAHV_walk','LALV_walk','HALV_walk','City1_Start','Rest_Start','Hwy_Start', 'City2_Start', 'City2_Start.1', 'Hwy_Start.1','City1_Start.1',  ]
    
    
    all_datas,all_label,all_user,all_task,all_quality=read_pickle(modality_name)

    control=[   "0001","0002","0003","0004","0005","0006","0007","0008","0009","0010","0011","0012","0013","0014","0015","0016","xianfei",'kirat', 'lohith7', 'lorn', 'lydia7',
 'missy', 'moira2', 'ruipeng', 'sawin', 'shaily5', 'shocky', 'susan8', 'tabitha5','timothy', 'vicki', 'xianfei', 'yuhsun', 'yuxuan',  'gabrielle', 'coung','georgie', 'hannah','avery' ,'brian', 'carolyn','christine','huaiyu', 'jingyu','jordan', 'joshua', 'kaite', 'kara8']
       
    oud=["8803","8804","8814","8829","8832","8840","8847","8852","8876","8898","9913","9915","9915v2","9925","9929","9933","9933v2","9941","9945","9945v2","9948","9952","9956","9967","9969","9973","9973v2","9984","9991"]
    
    predose=['8803','8804','8814','8829','8832','8840','8847','8876','8898','9913','9915v2','9933v2','9945v2','9973v2']
    postdose=['8852','9915','9925','9929','9933','9941','9945','9948','9952','9956','9967','9969','9973','9984','9991']

       
    #exercise=["P01","P02","P03","P04","P05","P06","P07","P08","P09","P10","P11","P12","P13","P15","P16","P17"]
    wesad=['S10' ,'S11','S13' ,'S14', 'S15', 'S16', 'S17', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9']
    
    alcohole=["Part101C","Part102C","Part104C","Part105C","Part106C","Part107C","Part108C","Part109C","Part110C","Part111C","Part112C"]
       
    control_v2=[ "alenav2","anjaliv2","brianav2","domv2","fabricv2","hanav2","jamiev2","karav2","kristinav2","nanv2","rubyv2","tomv2","xinhaov2"]
    
    
    emo=['1-E4-Drv1' ,'10-E4-Drv10',
 '11-E4-Drv11' ,'12-E4-Drv12' ,'13-E4-Drv13' ,'3-E4-Drv3' ,'4-E4-Drv4',
 '5-E4-Drv5' ,'6-E4-Drv6', '7-E4-Drv7' ,'8-E4-Drv8']

    test_control=['0006' ,'jordan', 'vicki','susan8','shaily5','0008','carolyn','0002','0016','avery','brian','0007','missy','0003']
    
    
    
    control=['user_'+x for x in control]
    oud=['user_'+x for x in oud]
    predose=['user_'+x for x in predose]
    postdose=['user_'+x for x in postdose]
    #exercise=['user_'+x for x in exercise]
    wesad=['user_'+x for x in wesad]
    alcohole=['user_'+x for x in alcohole]
    control_v2=['user_'+x for x in control_v2]
    emo=['user_'+x for x in emo]
    test_control=['user_'+x for x in test_control]

    
    train_users=list(set(control)-set(test_control))
   

    task_idx=np.argwhere(np.isin(all_task, task_set)).ravel()


    quality_idx=np.argwhere(all_quality<20)

    user_idx=np.argwhere(np.isin(all_user, train_users)).ravel()

    select_idx=reduce(np.intersect1d, [user_idx, task_idx, quality_idx])

    train_label=all_label[select_idx]
    train_label=list(map(lambda x: label_dict[x], train_label))
    train_user=all_user[select_idx]
    train_task=all_task[select_idx] 
    train_data=all_datas[select_idx]
    
    if norm_name=='minmaxChange':
        train_data,train_user,train_label,train_task=utils.user_changeScore(train_data,train_user,train_label,train_task)
    if norm_name=='standardChange':
        train_data,train_user,train_label,train_task=utils.user_changeScore2(train_data,train_user,train_label,train_task)
    elif norm_name=='standardUser':
        train_data=utils.user_standar(train_data,train_user,train_label)
    elif norm_name=='minmaxUser':
        train_data=utils.user_minmax(train_data,train_user,train_label)
       
    train_data=torch.Tensor(train_data)
    train_label=torch.Tensor(train_label)
    
    unique,unique_counts=np.unique(train_label,return_counts=True)
    unique=unique.astype(int)
    weight_per_class=[0]*len(unique)
    for i in unique:
        weight_per_class[i] = len(train_label)/unique_counts[i]
    weight = [0] * len(train_label)                                              
    for idx, val in enumerate(train_label): 
        weight[idx] = weight_per_class[int(val.item())] 
        
        
    irm_sampler=utils.RandomDomainSampler(train_user,50,10)
        
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weight, len(weight)) 

    dataset = H5Dataset(train_data, train_label,train_task,train_user,'train')
    
    if irm_batch:
        train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                            num_workers=4,
                                            batch_size=50,
                                            sampler=irm_sampler,
                                           drop_last=True,
                                           )
    else:
        train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                            num_workers=4,
                                            batch_size=50,
                                            sampler=sampler,
                                           drop_last=True,
                                           )

      
    loader_list=[]
    user_list=[test_control,wesad,predose,postdose,emo]
    
    name_list=['control','wesad','predose','postdose','emo']
    
    
    for name,user in zip(name_list,user_list):
        
        user_idx=np.argwhere(np.isin(all_user, user)).ravel()
        select_idx=reduce(np.intersect1d, [user_idx, task_idx, quality_idx])
      
    
        test_label=all_label[select_idx]
        test_label=list(map(lambda x: label_dict[x], test_label))
        test_user=all_user[select_idx]
        test_task=all_task[select_idx] 
    
        test_data=all_datas[select_idx]
        
        
        if norm_name=='minmaxChange':
            test_data,test_user,test_label,test_task=utils.user_changeScore(test_data,test_user,test_label,test_task)
        elif norm_name=='standardChange':
            test_data,test_user,test_label,test_task=utils.user_changeScore2(test_data,test_user,test_label,test_task)
           
        elif norm_name=='standardUser':
            
            test_data=utils.user_standar(test_data,test_user,test_label)
           
        elif norm_name=='minmaxUser':
            
            test_data=utils.user_minmax(test_data,test_user,test_label)
        
        test_data=torch.Tensor(test_data)
        test_label=torch.Tensor(test_label)

        dataset = H5Dataset(test_data, test_label,test_task,test_user,name)
        test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                            num_workers=4,
                                            batch_size=batch_size,
                                            shuffle=False,
                                          
                                       )
        loader_list.append(test_loader)
   
        
    return train_loader,loader_list


#get_dataloader('standardChange','eda')
    

    
