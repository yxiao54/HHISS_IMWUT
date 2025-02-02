import torch.nn.utils.prune as prune
import torch
import torch.nn as nn
import numpy as np
from torch.nn import CrossEntropyLoss
import random
import torch.utils.data as data_utils
import torch.nn.functional as F
import torch.autograd as autograd

class PruneDataset(torch.utils.data.Dataset):

    def __init__(self, data,label,user,task):
        self.data=data
        self.label=label
        self.task=task
        self.user=user
    
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
 
        return self.data[index],self.label[index],self.user[index],self.task[index]
        
        
def count_parameters(model):
    mydict = model.state_dict()
    layer_names = list(mydict)
    total_weights = 0
    non_zero_parameters = 0
    for i in layer_names:
        #print(i)
        if "weight" in i:
            weights = np.abs((model.state_dict()[i]).detach().cpu().numpy())
            total_weights += np.sum(np.ones_like(weights))
            non_zero_parameters += np.count_nonzero(weights)

    return non_zero_parameters/total_weights

def count_total_parameters(model):
    mydict = model.state_dict()
    layer_names = list(mydict)
    total_weights = 0
    for i in layer_names:
        if "weight" in i:
            weights = np.abs((model.state_dict()[i]).detach().cpu().numpy())
            total_weights += np.sum(np.ones_like(weights))
    return total_weights

def sum_total_parameters(model):
    mydict = model.state_dict()
    layer_names = list(mydict)
    total_weights = 0
    for i in layer_names:
        if "weight" in i:
            weights = np.abs((model.state_dict()[i]).detach().cpu().numpy())
            total_weights += np.sum(weights)
    return total_weights
    
def count_parameters_per_layer(model):
    mydict = model.state_dict()
    layer_names = list(mydict)
    total_weights = 0
    non_zero_parameters = 0
    layer_num = 1
    for i in layer_names:
        if "weight" in i:
            #print(i)
            weights = np.abs((model.state_dict()[i]).detach().cpu().numpy())
            total_weights = np.sum(np.ones_like(weights))
            non_zero_parameters = np.count_nonzero(weights)
            #print("Pruning at layer {}: {}".format(layer_num,(total_weights-non_zero_parameters)/total_weights))
            layer_num += 1
def calculate_l2_norm(model):
    l2_norm = 0.0
    for param in model.parameters():
        l2_norm += torch.sum(param ** 2)
    return torch.sqrt(l2_norm)
    
def calculate_l1_norm(model):
    l1_norm = 0.0
    for param in model.parameters():
        l1_norm += torch.sum(torch.abs(param))
    return l1_norm
    
def prune_model_structured_erdos_renyi_kernel(model,prune_amount):

    parameters_to_prune = []
    for module in  model.modules():
        if isinstance(module, nn.Conv2d):
            #print(module.kernel_size )
            scale = 1.0 - (module.in_channels + module.out_channels + module.kernel_size[0] + module.kernel_size[1] )/(module.in_channels * module.out_channels * module.kernel_size[0] * module.kernel_size[1])
            #print(scale)
            parameters_to_prune.append(scale * prune_amount)
            #parameters_to_prune.append(scale * prune_amount/2.0)
        elif (isinstance(module, nn.Linear) and module.out_features == 10):
            parameters_to_prune.append(prune_amount/2.0)
        elif isinstance(module, nn.Linear) :
            scale = 1.0 - (module.in_features + module.out_features)/(module.in_features * module.out_features)
            #print(scale)
            if prune_amount < 0.98 : parameters_to_prune.append(scale * prune_amount + 0.02)
            else: parameters_to_prune.append(scale * prune_amount)
    
    return parameters_to_prune



def compute_mask(module, score, prune_ratio):
    split_val = torch.quantile(score,prune_ratio)
    #print("----------------------")
    #print(split_val)
    #print("----------------------")
    struct_mask = torch.where(score <= split_val, 0.0,1.0)
    fine_mask_l = []
    
    weight = module.weight
        
    for mask, m in zip(struct_mask, weight):
        if mask == 0: 
            fine_mask_l.append(torch.zeros_like(m))
        else:
            fine_mask_l.append(torch.ones_like(m))
    #fine_mask_l = torch.cat(fine_mask_l,1)
    fine_mask = torch.stack(fine_mask_l)
    #print(fine_mask)
    #print(module.weight)
    #print(module.weight * fine_mask)
    return fine_mask,struct_mask








def magnitude_prune_unstruct_reverse(optimizer,model,loader,prune_amount, combine = "norm", prev_mask = None, bias = False,device = "cuda"):
    
    criterion = CrossEntropyLoss()
    importance_score = []
    batch = 0
    optimizer.zero_grad()
    for module in model.modules():
        if isinstance(module, nn.Conv2d)or isinstance(module, nn.Linear):
            importance_score.append(torch.abs(module.weight))
            #importance_score.append(torch.norm(module.weight,p=1,keepdim=True))
   
    #print(importance_score)
    fine_mask = []
    struct_mask = []
    masked_score = []
    score = []
    i = 0
    for imp_score in importance_score:
        if i == 0: score = torch.reshape(imp_score.data, [-1])
        else : score = torch.cat((score,torch.reshape(imp_score.data, [-1])),0)
        i += 1
    split_val = torch.quantile(score,prune_amount)
    for imp_score in importance_score:
        fine_mask.append(torch.where(imp_score <= split_val, 1.0,0.0))    

    #torch.where(importance_score3[2] < split_val, 0.0, importance_score3[2].double())
    #model = apply_mask(model, fine_mask,struct_mask)
    #count_parameters(model)
    #print(struct_mask)
    #print(fine_mask)
    return fine_mask, struct_mask, importance_score


def gradient_prune_unstruct1(trade_off,teacher_trade_off,optimizer,model,loader,prune_amount, loss_calculator,combine = "norm", prev_mask = None, bias = False,device = "cuda"):
    
    criterion = CrossEntropyLoss()
    importance_score = []
    batch = 0
    optimizer.zero_grad()
    for batch_idx, (inputs, targets,user,task) in enumerate(loader):
        batch += 1
        inputs, targets = inputs.to(device), targets.type(torch.LongTensor).to(device)
        outputs = model(inputs)
        outputs = outputs.type(torch.FloatTensor).to(device)
        loss = loss_calculator(inputs,outputs,targets,user,task,trade_off,teacher_trade_off)
        
        #loss += 0.0001 * calculate_l1_norm(model)
        loss += 0.0005 * calculate_l2_norm(model)
    
        loss.backward()
    
        batch_score = []
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                b_score = module.weight.grad
                
                
                batch_score.append(b_score)
        if batch == 1:
            importance_score = batch_score
        else:
            sum_is = []
            for i,b in zip (importance_score,batch_score):
                sum_is.append(i+b)
            importance_score = sum_is
    #print(importance_score)
    i = 0        
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            importance_score[i] = torch.abs(module.weight * importance_score[i])
            #importance_score[i] = torch.abs(importance_score[i])
            i += 1
    
    #print(importance_score)
    fine_mask = []
    struct_mask = []
    masked_score = []
    score = []
    i = 0
    for imp_score in importance_score:
        if i == 0: score = torch.reshape(imp_score.data, (-1,))
        else : score = torch.cat((score,torch.reshape(imp_score.data, (-1,))),0)
        i += 1
    split_val = torch.quantile(score,prune_amount)
    for imp_score in importance_score:
        fine_mask.append(torch.where(imp_score <= split_val, 0.0,1.0))    
    """
    for m in fine_mask:
        print(m.size())
    """
    #torch.where(importance_score3[2] < split_val, 0.0, importance_score3[2].double())
    #model = apply_mask(model, fine_mask,struct_mask)
    count_parameters(model)
    #print(struct_mask)
    #print(fine_mask)
    return fine_mask, struct_mask, importance_score




def apply_mask(model, model_mask,struct_mask = None):
    mydict = model.state_dict()
    layer_names = list(mydict)
    
    #print(layer_names)
    #print(model_mask)
    #print(struct_mask)
    i = 0
    if "weight" in layer_names[0]: 
        w_ln = 0
        b_ln = 1
    else: 
        w_ln = 1
        b_ln = 0
    for module in model.modules():
        #print(layer_names[w_ln])
        if "bias" in layer_names[w_ln]:
            w_ln = w_ln +1
            b_ln = b_ln +1
        else:
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d) or (isinstance(module, nn.Linear) and module.out_features != 2) and 'mask' not in layer_names[w_ln]:
                mask = model_mask[i]
                #print(layer_names[w_ln])
                
                #print(model.state_dict()[layer_names[w_ln]].shape)
                model.state_dict()[layer_names[w_ln]].copy_(module.weight * mask)
                i = i + 1
                w_ln = w_ln+1
                b_ln = b_ln+1
            elif isinstance(module, nn.BatchNorm2d):
                w_ln = w_ln+5
                b_ln = b_ln+5
    return model

    
def sum_total_parameters(model):
    mydict = model.state_dict()
    layer_names = list(mydict)
    total_weights = 0
    for i in layer_names:
        if "weight" in i:
            weights = np.abs((model.state_dict()[i]).detach().cpu().numpy())
            #print(weights)
            total_weights += np.sum(weights)
    return total_weights


def forward_hook_fn(module, input, output):
    if isinstance(module,nn.Conv2d):
        weight = module.weight * module.mask
        output = F.conv2d(
            input[0], weight, module.bias, module.stride, module.padding, module.dilation, module.groups
        )
        #print("conv hook implemeted")
    elif isinstance(module,nn.Linear):
        weight = module.weight * module.mask
        output = F.linear(input[0], weight, module.bias)
        #print("linear hook implemeted")
        
    return output

def mask_forward_only(model, model_mask):
    i = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d) or (isinstance(module, nn.Linear) and module.out_features != 2):
            module.register_buffer('mask', model_mask[i])
            i += 1
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d) or (isinstance(module, nn.Linear) and module.out_features != 2):
            module.register_forward_hook(forward_hook_fn)

    
def Prune(trade_off,teacher_trade_off,optimizer,model, all_data,all_label,all_user,all_task, prune_amount, prune_mechanism,loss_calculator, device = "cuda",):
    
    parameters_to_prune = prune_model_structured_erdos_renyi_kernel(model,prune_amount)
    
    count_parameters(model)
    
    
    
    
  
   
    if prune_mechanism in ['erm_gradient','irm_gradient','vrex_gradient','dro_gradient']:
        train_set = PruneDataset(all_data, all_label,all_user,all_task)
        data_loader = data_utils.DataLoader(train_set, batch_size=50, shuffle=True)
        model_mask, struct_mask, masked_score = gradient_prune_unstruct1(trade_off,teacher_trade_off,optimizer,model,data_loader,prune_amount,loss_calculator,device = device)
        model = apply_mask(model, model_mask,struct_mask) 
        mask_forward_only(model, model_mask)
    elif prune_mechanism in ['erm_weight','irm_weight','vrex_weight','dro_weight']:
        train_set = PruneDataset(all_data, all_label,all_user,all_task)
        data_loader = data_utils.DataLoader(train_set, batch_size=50, shuffle=True)
        model_mask, struct_mask, masked_score = magnitude_prune_unstruct_reverse(optimizer,model,data_loader,prune_amount,device = device)
        model = apply_mask(model, model_mask,struct_mask) 
        mask_forward_only(model, model_mask)
    elif prune_mechanism in ['HHISS','ourIRM','ourDRO','ourVrex']:
        unique=np.unique(all_user)
        
        all_masks=[]
        for user in unique:
            user_idx=np.argwhere(np.isin(all_user, [user])).ravel()
            data=all_data[user_idx]
            label=all_label[user_idx]
            current_user=all_user[user_idx]
            current_task=all_task[user_idx]
            
            train_set = PruneDataset(data, label,current_user,current_task)
            data_loader = data_utils.DataLoader(train_set, batch_size=50, shuffle=True)
            model_mask, struct_mask, masked_score = gradient_prune_unstruct1(trade_off,teacher_trade_off,optimizer,model,data_loader,prune_amount,loss_calculator,device = device)
            all_masks.append(model_mask)
        
        model_mask=all_masks[0]
        for mask in all_masks:
     
            for i,layer in enumerate(mask):

                model_mask[i]=model_mask[i].int()&layer.int()

        model = apply_mask(model, model_mask,struct_mask) 
    elif prune_mechanism in ['taskIRM','taskDRO','taskVrex']:
        unique=np.unique(all_task)
        
        all_masks=[]
        for task in unique:
            task_idx=np.argwhere(np.isin(all_task, [task])).ravel()
            data=all_data[task_idx]
            label=all_label[task_idx]
            current_user=all_user[task_idx]
            current_task=all_task[task_idx]
            train_set = PruneDataset(data, label,current_user,current_task)
            data_loader = data_utils.DataLoader(train_set, batch_size=50, shuffle=True)
            model_mask, struct_mask, masked_score = gradient_prune_unstruct1(trade_off,teacher_trade_off,optimizer,model,data_loader,prune_amount,loss_calculator,device = device)
            all_masks.append(model_mask)
        
        model_mask=all_masks[0]
        for mask in all_masks:
     
            for i,layer in enumerate(mask):

                model_mask[i]=model_mask[i].int()&layer.int()

        model = apply_mask(model, model_mask,struct_mask) 
        
    elif prune_mechanism in ['hybridIRM','hybridDRO','hybridVrex']:
        
        unique_user=np.unique(all_user)
        unique_task=np.unique(all_task)
        
        all_masks=[]
        for user in unique_user:
            user_idx=np.argwhere(np.isin(all_user, [user])).ravel()
            data=all_data[user_idx]
            label=all_label[user_idx]
            current_user=all_user[user_idx]
            current_task=all_task[user_idx]
            train_set = PruneDataset(data, label,current_user,current_task)
            data_loader = data_utils.DataLoader(train_set, batch_size=50, shuffle=True)
            model_mask, struct_mask, masked_score = gradient_prune_unstruct1(trade_off,teacher_trade_off,optimizer,model,data_loader,prune_amount,loss_calculator,device = device)
            all_masks.append(model_mask)
        for task in unique_task:
            task_idx=np.argwhere(np.isin(all_task, [task])).ravel()
            data=all_data[task_idx]
            label=all_label[task_idx]
            current_user=all_user[task_idx]
            current_task=all_task[task_idx]
            train_set = PruneDataset(data, label,current_user,current_task)
            data_loader = data_utils.DataLoader(train_set, batch_size=50, shuffle=True)
            model_mask, struct_mask, masked_score = gradient_prune_unstruct1(trade_off,teacher_trade_off,optimizer,model,data_loader,prune_amount,loss_calculator,device = device)
            all_masks.append(model_mask)
        
        model_mask=all_masks[0]
        for mask in all_masks:
     
            for i,layer in enumerate(mask):

                model_mask[i]=model_mask[i].int()&layer.int()

        model = apply_mask(model, model_mask,struct_mask)    
    elif prune_mechanism in ['sparseTrain']:
        train_set = PruneDataset(all_data, all_label,all_user,all_task)
        data_loader = data_utils.DataLoader(train_set, batch_size=50, shuffle=True)
        model_mask, struct_mask, masked_score = gradient_prune_unstruct1(trade_off,teacher_trade_off,optimizer,model,data_loader,prune_amount,loss_calculator,device = device)
        model = apply_mask(model, model_mask,struct_mask) 
     
       
    


    return model, model_mask
   
    
