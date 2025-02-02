import numpy as np
import pandas as pd
import scipy.io
import glob
import neurokit2 as nk
from scipy import signal
import pickle
import tqdm
from datetime import datetime
import time
from scipy.stats import skew,kurtosis 
import os
import flirt


summarize=[]

save_eda=[]
save_ppg=[]
save_acc=[]
save_hr=[]
save_temp=[]

save_hrv=[]
save_tonic=[]
save_phasic=[]

save_label=[]
save_user=[]
save_task=[]

save_quality=[]

error=0

datas=['/scratch/yxiao124/data/oud.pickle','/scratch/yxiao124/data/control.pickle']
progress= open("full_progress.txt", "w")
for folder in datas:
    with open(folder,'rb') as hande:
        writeout = pickle.load(hande)
    all_eda=writeout["eda"]
    all_ppg= writeout["ppg"]
    all_hr=writeout["hr"]
    all_temp=writeout["temp"]
    all_acc=writeout["acc"]
    all_label=writeout["label"]
    all_user=writeout["user"]
    all_task=writeout["task"]
    total_len=len(all_eda)
    counter=0
    progress.write("{} \n".format(folder))
    progress.write("{} \n".format(total_len))
    
    
    progress.flush()
    
    for data_eda,data_ppg,data_hr,data_temp,data_acc,data_label,data_user,data_task in zip(all_eda,all_ppg,all_hr,all_temp,all_acc,all_label,all_user,all_task):
        
        counter+=1
        progress.write("{}% \n".format(counter/total_len))
        progress.flush()
    
        

        try:
            data_eda = nk.signal_resample(data_eda, sampling_rate=4, desired_sampling_rate=64, method='poly')
            data_eda = nk.eda_clean(data_eda, sampling_rate=64,method='neurokit')
            data_eda, _ = nk.eda_process(data_eda, sampling_rate=64,method='neurokit')
            data_ppg, _ = nk.ppg_process(data_ppg, sampling_rate=64)
        except:
            print('preprocesss error')
            continue
        
        sr_eda=64
        sr_ppg=64                                                                                         
        sr_temp=4
        sr_hr=1
        sr_acc=32
        sr_clean=64
           
           
    
        data_x=data_acc[:,0]
        data_y=data_acc[:,1]
        
        data_z=data_acc[:,2]
            
        all_data=[]
        length=((len(data_hr))-15)/15
        print(length)
        print(len(data_hr))
        step=15
        window=30
        
        for i in range(int(length)):
            try:
                start=i*step
                end=i*step+window
                
           
                temp_ppg=data_ppg.iloc[start*sr_ppg:end*sr_ppg]
                temp_temp=data_temp[start*sr_temp:end*sr_temp]
                temp_hr=data_hr[start*sr_hr:end*sr_hr]
                
                temp_x=data_x[start*sr_acc:end*sr_acc]
                temp_y=data_y[start*sr_acc:end*sr_acc]
                temp_z=data_z[start*sr_acc:end*sr_acc]
    
     
                data_signal=data_eda.iloc[start*sr_eda:end*sr_eda]
        
                tonic=data_signal['EDA_Tonic']
                phasic=data_signal['EDA_Phasic']
                
                num_onset=data_signal['SCR_Onsets']
                num_peaks=data_signal['SCR_Peaks']
                num_recovery=data_signal['SCR_Recovery']
                
                height=data_signal['SCR_Height']
                height = height[height!=0]
                
                amplitude=data_signal['SCR_Amplitude']
                
                amplitude = amplitude[amplitude!=0]
        
                risetime=data_signal['SCR_RiseTime']
                risetime = risetime[risetime!=0]
                recoverytime=data_signal['SCR_RecoveryTime']
                recoverytime = recoverytime[recoverytime!=0]
    
                filtered =temp_ppg['PPG_Clean']
    
                
                
                quality_res=nk.ppg_quality(filtered,sampling_rate=64,method='disimilarity')
                quality_res=quality_res.reshape(-1)
                
                hrv_time=nk.hrv_time(temp_ppg,sampling_rate=64,psd_method="lomb").to_numpy().reshape(-1)
                hrv_frequency=nk.hrv_frequency(temp_ppg,sampling_rate=64,psd_method="lomb").to_numpy().reshape(-1)
                
                hrv_feature=np.concatenate([hrv_time,hrv_frequency])
           
             
                eda_feature=np.stack([np.sum(num_onset),np.sum(num_peaks),np.sum(num_recovery),np.mean(height),np.mean(amplitude),np.mean(risetime),np.mean(recoverytime)]).reshape(-1)
                
                
            
                
                hr=pd.DataFrame({'HR' : temp_hr})
                temp=pd.DataFrame({'TEMP' : temp_temp})
                acc=pd.DataFrame({'x' : temp_x,'y' : temp_y,'z' : temp_z})
                bvp=pd.DataFrame({'BVP' : filtered})
                
                tonic=pd.DataFrame({'tonic' : tonic})
                phasic=pd.DataFrame({'phasic' : phasic})
                
                
               
            
                acc = acc.set_index( pd.date_range(start=0, periods=len(acc), freq='31250000N'))
                acc = acc.fillna(0)
                acc_features=flirt.get_acc_features(acc,window_length=window,window_step_size=window, data_frequency=32)
                acc_features=acc_features.add_prefix('ACC_').to_numpy().reshape(-1)
                
                temp = temp.set_index( pd.date_range(start=0, periods=len(temp), freq='250000000N'))
                temp = temp.fillna(0)
                temp_features=flirt.get_acc_features(temp,window_length=window,window_step_size=window, data_frequency=4)
                temp_features=temp_features.add_prefix('TEMP_').to_numpy().reshape(-1)
                
         
                hr = hr.set_index( pd.date_range(start=0, periods=len(hr), freq='1S'))
                hr = hr.fillna(0)
                hr_features=flirt.get_acc_features(hr,window_length=window,window_step_size=window, data_frequency=1)
                hr_features=hr_features.add_prefix('HR_').to_numpy().reshape(-1)
                
             
                bvp = bvp.set_index( pd.date_range(start=0, periods=len(bvp), freq='15625000N'))
                bvp = bvp.fillna(0)
                bvp_features=flirt.get_acc_features(bvp,window_length=window,window_step_size=window, data_frequency=64)
                bvp_features=bvp_features.add_prefix('BVP_').to_numpy().reshape(-1)
                
                tonic = tonic.set_index( pd.date_range(start=0, periods=len(tonic), freq='15625000N'))
                tonic = tonic.fillna(0)
                tonic_features=flirt.get_acc_features(tonic,window_length=window,window_step_size=window, data_frequency=64)
                tonic_features=tonic_features.add_prefix('Tonic_').to_numpy().reshape(-1)
                
                phasic = phasic.set_index( pd.date_range(start=0, periods=len(phasic), freq='15625000N'))
                phasic = phasic.fillna(0)
                phasic_features=flirt.get_acc_features(phasic,window_length=window,window_step_size=window, data_frequency=64)
                
                
                
                phasic_features=phasic_features.add_prefix('Phasic_').to_numpy().reshape(-1)
    
       
                save_eda.append(eda_feature)
                save_ppg.append(bvp_features)
                save_acc.append(acc_features)
                save_hr.append(hr_features)
                save_temp.append(temp_features)
                
                save_hrv.append(hrv_feature)
                save_tonic.append(tonic_features)
                save_phasic.append(phasic_features)
                
                save_label.append(data_label)
                save_user.append(data_user)
                save_task.append(data_task)
                save_quality.append(quality_res)
               
            except:
                print('extract error')
                error+=1
                
                continue
progress.close()
save_eda=np.asarray(save_eda)
save_ppg=np.asarray(save_ppg)
save_hr=np.asarray(save_hr)
save_temp=np.asarray(save_temp)
save_acc=np.asarray(save_acc)
save_hrv=np.asarray(save_hrv)
save_tonic=np.asarray(save_tonic)
save_phasic=np.asarray(save_phasic)


print(error) 
writeout={}
writeout["eda"]=save_eda
writeout["ppg"]=save_ppg
writeout["hr"]=save_hr
writeout["temp"]=save_temp
writeout["acc"]=save_acc

writeout["hrv"]=save_hrv
writeout["tonic"]=save_tonic
writeout["phasic"]=save_phasic


writeout["label"]=save_label
writeout["user"]=save_user
writeout["task"]=save_task
writeout["quality"]=save_quality

with open('./data/full_drive.pickle', 'wb') as handle:
    pickle.dump(writeout, handle)
   
    
   
    