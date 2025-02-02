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
    def __init__(self,model_train,optimizer_train,approach_name,filename,device,hidden_size,norm_name,input_dim,modality):
        self.model=model_train
        self.device=device
        self.optimizer=optimizer_train
  
    

        self.filename=filename
        self.cross_entropy=nn.CrossEntropyLoss()
      
        self.get_loss = mylosses.IRM(device)
 
        
     

                
   
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
        
    

def objective(filename,approach_name,hidden_name,norm_name,modality_name):

    params={
            "LR":0.0001,
             
              "drop":0.3,
              
              
    }
    setseed(364)

    train_loader,val_loader=get_dataloader(norm_name,modality_name,irm_batch=False)
  
    hidden_size=hidden_name
    
    
    input_dim=train_loader.dataset.data.shape[-1]
    model_train = Baseline(hidden=hidden_size,drop=params["drop"],input_dim=input_dim,num_class=2)
    
    device_train = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_train.to(device_train)

    optimizer_train = optim.Adam(model_train.parameters(), lr=params["LR"])#
    


    
    eng =Engine(model_train,optimizer_train,approach_name,filename,device_train,hidden_size,norm_name,input_dim,modality_name)
    
    
    
    epoch=100
    early_stop=30
    best_acc_dict={'control':0,'wesad':0,'predose':0,'postdose':0,'affectiveRoad':0}
    best_accumulate_acc=0
    


    for e in range(epoch):
        
        train_f1=eng.train(train_loader)
      

        temp_acc_dict={'control':0,'wesad':0,'predose':0,'postdose':0,'affectiveRoad':0}
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

 
    return best_accumulate_acc
       
    
if __name__ == '__main__':

  
    
    approach_name='irm'
    norm_name='standardChange'
    hidden_name=256
    modality_name="all"

    path=f"./Teacher/{approach_name}/{norm_name}/{hidden_name}/{modality_name}"
    if not os.path.exists(path):
        os.makedirs(path)
    filename=path+"/log.txt"
    with open(filename, 'w') as f:
        print("prgram start",file=f)
    
    objective(filename,approach_name,hidden_name,norm_name,modality_name)
    
   

    
