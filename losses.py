import torch
from torch import nn


import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np
from models import Baseline

class InvariancePenaltyLoss(nn.Module):

    def __init__(self):
        super(InvariancePenaltyLoss, self).__init__()
        self.scale = torch.tensor(1.).requires_grad_()

    def forward(self, y: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss_1 = F.cross_entropy(y[::2] * self.scale, labels[::2])
        loss_2 = F.cross_entropy(y[1::2] * self.scale, labels[1::2])
        grad_1 = autograd.grad(loss_1, [self.scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [self.scale], create_graph=True)[0]
        penalty = torch.sum(grad_1 * grad_2)
        return penalty
class AutomaticUpdateDomainWeightModule(object):
    def __init__(self, num_domains: int, eta: float, device):
        self.domain_weight = torch.ones(num_domains).to(device) / num_domains
        self.eta = eta

    def get_domain_weight(self, sampled_domain_idxes):
      
        domain_weight = self.domain_weight[sampled_domain_idxes]
        domain_weight = domain_weight / domain_weight.sum()
        return domain_weight

    def update(self, sampled_domain_losses: torch.Tensor, sampled_domain_idxes):
      
        sampled_domain_losses = sampled_domain_losses.detach()

        for loss, idx in zip(sampled_domain_losses, sampled_domain_idxes):
            self.domain_weight[idx] *= (self.eta * loss).exp()
            
class ERM(nn.Module):

    def __init__(self):
        super(ERM, self).__init__()
      
        self.loss_class=nn.CrossEntropyLoss()
        

    def forward(self,data,outputs,label,user,task,trade_off,teacher_trade_off):
        loss = self.loss_class(outputs, label)

        return loss
class IRM(nn.Module):

    def __init__(self,device):
        super(IRM, self).__init__()
        self.invariance_penalty_loss = InvariancePenaltyLoss().to(device)
       
        self.loss_class=nn.CrossEntropyLoss()
        

    def forward(self,data,outputs,label,user,task,trade_off,teacher_trade_off):
        
        loss_ce= self.loss_class(outputs, label)

        unique=np.unique(user)
        loss_penalty = 0
        for u in unique:
            idx=np.argwhere(np.isin(user, [u]))
    
    
            idx=np.squeeze(idx,1)
            y_per_domain=outputs[idx]
            
    
            labels_per_domain=label[idx]
            n_domains_per_batch=len(y_per_domain)
            
            loss_penalty += self.invariance_penalty_loss(y_per_domain, labels_per_domain) / n_domains_per_batch
        
        
        loss = loss_ce + loss_penalty*trade_off

        
            
        return loss
class Vrex(nn.Module):

    def __init__(self,device):
        super(Vrex, self).__init__()
        self.device=device
        
        

    def forward(self,data,outputs,label,user,task,trade_off,teacher_trade_off):
        
        

        unique=np.unique(user)
        loss_penalty = 0
        loss_ce_per_domain = torch.zeros(len(unique)).to(self.device)
        for domain_id,u in enumerate(unique):
            idx=np.argwhere(np.isin(user, [u]))
            idx=np.squeeze(idx,1)
            y_per_domain=outputs[idx]
            labels_per_domain=label[idx]
        
            loss_ce_per_domain[domain_id] = F.cross_entropy(y_per_domain, labels_per_domain)

        loss_ce = loss_ce_per_domain.mean()
    
        loss_penalty = ((loss_ce_per_domain - loss_ce) ** 2).mean()
        
        loss = loss_ce + loss_penalty*trade_off
        
            
        return loss
        
class Dro(nn.Module):

    def __init__(self,all_user,device):
        super(Dro, self).__init__()
        
        self.num_user=len(all_user)
        self.domain_weight_module = AutomaticUpdateDomainWeightModule(self.num_user, 1e-2, device)
        self.get_idx = {x: i for i, x in enumerate(all_user)}
        self.device=device

    def forward(self,data,outputs,label,user,task,trade_off,teacher_trade_off):
        
        

        unique=np.unique(user)
        loss_per_domain = torch.zeros(len(unique)).to(self.device)
        for domain_id,u in enumerate(unique):
            idx=np.argwhere(np.isin(user, [u]))
            idx=np.squeeze(idx,1)
            y_per_domain=outputs[idx]
            labels_per_domain=label[idx]
            
            
            loss_per_domain[domain_id] = F.cross_entropy(y_per_domain, labels_per_domain)
           
            
        sampled_domain_idxes=[self.get_idx[u] for u in unique]
        # update domain weight
        self.domain_weight_module.update(loss_per_domain, sampled_domain_idxes)
        domain_weight = self.domain_weight_module.get_domain_weight(sampled_domain_idxes)

        # weighted cls loss
        loss = (loss_per_domain * domain_weight).sum()
            
        return loss
        
    
              
class HHISS(nn.Module):

    def __init__(self,hidden_size,device,norm_name,input_dim,modality,teacher=True):
        super(HHISS, self).__init__()
        self.invariance_penalty_loss = InvariancePenaltyLoss().to(device)
        
        self.device=device
        self.teacher=teacher
    
        if teacher:
            
            self.model_teacher = Baseline(hidden=hidden_size,drop=0.2,input_dim=input_dim,num_class=2)
            self.model_teacher.to(device)
    
            self.model_teacher.load_state_dict(torch.load(f'./ckpt/Overparameterized_IRM.pth', map_location='cpu'))
    
        self.loss_class=nn.CrossEntropyLoss()
        

    def forward(self,data,outputs,label,user,task,trade_off,teacher_trade_off):
        
        loss_ce= self.loss_class(outputs, label)

        unique=np.unique(user)
        loss_penalty = 0
        for u in unique:
            idx=np.argwhere(np.isin(user, [u]))
    
    
            idx=np.squeeze(idx,1)
            y_per_domain=outputs[idx]
            
    
            labels_per_domain=label[idx]
            n_domains_per_batch=len(y_per_domain)
            
            loss_penalty += self.invariance_penalty_loss(y_per_domain, labels_per_domain) / n_domains_per_batch
        if self.teacher:
            teacher_outputs = self.model_teacher(data).detach()
        
            loss = loss_ce + loss_penalty*trade_off+teacher_trade_off*self.loss_class(outputs,teacher_outputs)
        else:
            loss = loss_ce + loss_penalty*trade_off
        
            
        return loss
