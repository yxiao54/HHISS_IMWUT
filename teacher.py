import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.functional as F
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,balanced_accuracy_score
import numpy as np
import os
import optuna
import sys
import random
import shutil
import glob

from prune_utils import Prune,count_parameters
import torch.utils.data as data_utils
from function import  get_dataloader
import losses as mylosses

from models import Baseline
def setseed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def calculate_l2_norm(model):
    l2_norm = 0.0
    for param in model.parameters():
        l2_norm += torch.sum(param ** 2)
    return torch.sqrt(l2_norm)

class Engine:
    def __init__(self,model_train,optimizer_train,approach_name,filename,device,hidden_size,all_task,all_user,norm_name,input_dim,modality):
        self.model=model_train
        self.device=device
        self.optimizer=optimizer_train
  
    

        self.filename=filename
        self.cross_entropy=nn.CrossEntropyLoss()
        #'ourApproach','erm','irm','vrex','dro','ermPrune','irmPrune','sparseTrain','KD'
        
        if approach_name=='ourIRM':
            self.get_loss = mylosses.OurIRM(hidden_size,device,norm_name,input_dim,modality,teacher=False)
        elif approach_name=='ourDRO':
            self.get_loss = mylosses.OurDRO(all_user,hidden_size,device,norm_name,input_dim,teacher=False)
        elif approach_name=='ourVrex':
            self.get_loss = mylosses.OurVrex(hidden_size,device,norm_name,input_dim,teacher=False)
        elif approach_name=='taskIRM':
            self.get_loss = mylosses.TaskIRM(hidden_size,device,norm_name,input_dim,teacher=False)
        elif approach_name=='taskDRO':
            self.get_loss = mylosses.TaskDRO(all_task,hidden_size,device,norm_name,input_dim,teacher=False)
        elif approach_name=='taskVrex':
            self.get_loss = mylosses.TaskVrex(hidden_size,device,norm_name,input_dim,teacher=False)
        elif approach_name=='hybridIRM':
            self.get_loss = mylosses.HybridIRM(hidden_size,device,norm_name,input_dim,teacher=False)
        elif approach_name=='hybridDRO':
            self.get_loss = mylosses.HybridDRO(all_user,all_task,hidden_size,device,norm_name,input_dim,teacher=False)
        elif approach_name=='hybridVrex':
            self.get_loss = mylosses.HybridVrex(hidden_size,device,norm_name,input_dim,teacher=False)
            
        elif approach_name=='erm':
            self.get_loss = mylosses.ERM()
        elif approach_name=='irm':
            self.get_loss = mylosses.IRM(device)
        elif approach_name=='dro':
            self.get_loss = mylosses.Dro(all_user,device)
        elif approach_name=='vrex':
            self.get_loss = mylosses.Vrex(device)
      
        
        self.prune_name=prune_name

                
   
    def train(self,data_loader,trade_off):
        self.model.train()
        final_loss = 0
        truth=[]
        predict=[]
        teacher_trade_off=0
      
        for batch_idx, (data,label,task,user) in enumerate(data_loader):

            data,label = data.to(self.device).float(), label.to(self.device).long()
            data= Variable(data)
            

            outputs = self.model(data)
            y_hat = torch.max(outputs,1)[1]
     
            loss=self.get_loss(data,outputs,label,user,task,trade_off,teacher_trade_off)
            
            loss += 0.0005 * calculate_l2_norm(self.model)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            final_loss+= loss.item()
            truth.extend(label.tolist())
            predict.extend(y_hat.tolist())

        f1score=f1_score(truth,predict,average='macro')
        
        acc=accuracy_score(truth,predict)
        cf=confusion_matrix(truth,predict)
        with open(self.filename, 'a+') as handle:
            print('Train : Loss: {:.4f}, Train acc : {:.4f}, Train f1 : {:.4f}'.format(final_loss/len(data_loader),acc,f1score),file=handle)
        return f1score
    def evaluate(self,data_loader,eval_type):
        self.model.eval()
        
        final_loss = 0
        truth=[]
        predict=[]
        nameid=[]
        taskid=[]
        for batch_idx, (data,label,task,user) in enumerate(data_loader):

            data,label = data.to(self.device).float(), label.to(self.device).long()
            data= Variable(data)

            outputs = self.model(data)
            
            y_hat = torch.max(outputs,1)[1]
            loss = self.cross_entropy(outputs, label)

            final_loss+= loss.item()
            truth.extend(label.tolist())
            predict.extend(y_hat.tolist())
            nameid.append(user)
            taskid.append(task)
       
            
        f1score=f1_score(truth,predict,average='macro')
        
        acc=balanced_accuracy_score(truth,predict)
        cf=confusion_matrix(truth,predict)
        with open(self.filename, 'a+') as handle:
            print('{} : Loss: {:.4f}, val acc : {:.4f}, val f1 : {:.4f}'.format(eval_type,final_loss/len(data_loader),acc,f1score),file=handle)
            print(cf,file=handle)

        
        return f1score,acc
        
    

def objective(filename,approach_name,prune_name,amount_name,hidden_name,norm_name,modality_name,trial=None):

    params={
            "LR":trial.suggest_float("LR",1e-4,1e-3),
             
              "drop":0.3,
              'prune_threshold':trial.suggest_categorical("prune_threshold", [0.7,0.75,0.8,0.85])
              
    }
    setseed(364)

    train_loader,val_loader=get_dataloader(norm_name,modality_name,irm_batch=False)
    prune_amout=amount_name
    hidden_size=hidden_name
    
    
    input_dim=train_loader.dataset.data.shape[-1]
    model_train = Baseline(hidden=hidden_size,drop=params["drop"],input_dim=input_dim,num_class=2)
    
    device_train = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_train.to(device_train)
    #if approach_name in ['ourApproach','ermPrune','irmPrune']:
    #    model_train.load_state_dict(torch.load(f'./Teacher/{hidden_name}/Best.pth', map_location='cpu'))

    
    optimizer_train = optim.Adam(model_train.parameters(), lr=params["LR"])#
    
    all_user=[]
    all_task=[]
    
    for _,_,task,user in train_loader:
        all_user.append(user)
        all_task.append(task)
        
    all_user=np.concatenate(all_user)
    all_user=np.unique(all_user)
    all_task=np.concatenate(all_task)
    all_task=np.unique(all_task)

    
    eng =Engine(model_train,optimizer_train,approach_name,filename,device_train,hidden_size,all_task,all_user,norm_name,input_dim,modality_name)
    
    epoch=100
    early_stop=30
    best_acc_dict={'control':0,'control_v2':0,'wesad':0,'exercise':0,'predose':0,'postdose':0,'emo':0}
    best_accumulate_acc=0
    
    
    
    trade_off=1
    threshold=params['prune_threshold']

    for e in range(epoch):
        
        train_f1=eng.train(train_loader,trade_off)
        if e>30:
            trade_off=0.3

        temp_acc_dict={'control':0,'control_v2':0,'wesad':0,'exercise':0,'predose':0,'postdose':0,'emo':0}
        for loader in val_loader:
            dataset_name=loader.dataset.dataset_name
            val_f1,val_acc=eng.evaluate(loader,dataset_name)

            if train_f1>0.7:
                temp_acc_dict[dataset_name]=val_acc

            
        accumulate_acc=0
        for key in  temp_acc_dict:
            accumulate_acc+=temp_acc_dict[key]
        
        if accumulate_acc>best_accumulate_acc:
            torch.save(eng.model.state_dict(), filename.replace('log.txt','best'+str(accumulate_acc)+'_'+str(trial.number)+'.pth'))
            best_accumulate_acc=accumulate_acc
            best_acc_dict=temp_acc_dict
            
    for key in  best_acc_dict:
        acc=best_acc_dict[key]

        with open(filename, 'a+') as handle:
            print(f'best {key} acc:{acc:4f}',file=handle)

    with open(filename, 'a+') as handle:
        print('trail ',trial.number,'end_',params["LR"],file=handle)
    
    return best_accumulate_acc
       
    
if __name__ == '__main__':

    paras=sys.argv[1:]
    approach_idx=int(paras[0])

    prune_idx=0
    
    amount_idx=0
    hidden_idx=int(paras[1])
    norm_idx=int(paras[2])
    trial_num=3
    
    modality_idx=int(paras[3])
    
    
    all_modality=["all",'eda','edabvp','edahr','edahrv','edatemp','edaacc']
    modality_name=all_modality[modality_idx]

    
    all_approach=['erm','irm','vrex','dro','KD','sparseTrain','erm_weight','irm_weight','vrex_weight','dro_weight','erm_gradient','irm_gradient','vrex_gradient',
    'dro_gradient','ourIRM','ourDRO','ourVrex','taskIRM','taskDRO','taskVrex','hybridIRM','hybridDRO','hybridVrex']
    approach_name=all_approach[approach_idx]
    
    all_prune=['noPrune','weight','gradient','subjectWise']
    prune_name=all_prune[prune_idx]
    
    
    all_amount=[0.2,0.5,0.8]
    amount_name=all_amount[amount_idx]
    
    all_hidden=[64,128,256,512]
    hidden_name=all_hidden[hidden_idx]

   
    all_norm=['minmaxChange','standardChange']
    norm_name=all_norm[norm_idx]

    path=f"./Teacher/{approach_name}/{norm_name}/{hidden_name}/{modality_name}"
    if not os.path.exists(path):
        os.makedirs(path)
    filename=path+"/log.txt"
    with open(filename, 'w') as f:
        print("prgram start",file=f)
    
    sampler = optuna.samplers.TPESampler(seed=2023)
    study=optuna.create_study(sampler=sampler,direction="maximize",study_name="hyper")
    study.optimize(lambda trial: objective(filename,approach_name,prune_name,amount_name,hidden_name,norm_name,modality_name,trial),n_trials=trial_num)

    trial_=study.best_trial
    
    path=glob.glob(f"./Teacher/{approach_name}/{norm_name}/{hidden_name}/{modality_name}/*.pth")
    
    path=sorted(path)
    src=path[-1]
    dst=f"./Teacher/{approach_name}/{norm_name}/{hidden_name}/{modality_name}/best.pth"

    shutil.copyfile(src, dst)
    

    
