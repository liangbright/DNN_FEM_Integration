import sys
sys.path.append("./c3d8")
sys.path.append("./mesh")
import numpy as np
import os
import pandas as pd
from PolyhedronMesh import PolyhedronMesh
from von_mises_stress import cal_von_mises_stress
from principal_stress import cal_max_abs_principal_stress
from train_val_test_split_x_c_m import train_val_test_split
#%%
def compare(filelist_true, filstlist_pred, stress):
    MAPE_list=[]
    APE_list=[]
    mrse_mean_list=[]
    mrse_min_list=[]
    mrse_max_list=[]
    for file_true, file_pred in zip(filelist_true, filstlist_pred):
        #mesh_p0=PolyhedronMesh()
        #file_p0=file_true.replace('i18', 'i0')
        #mesh_p0.load_from_torch(file_p0+'.pt')
        mesh_true=PolyhedronMesh()
        mesh_true.load_from_torch(file_true+'.pt')
        mesh_pred=PolyhedronMesh()
        mesh_pred.load_from_torch(file_pred+'.pt')
        #---------------------------------------
        #X_true=mesh_p0.node
        x_true=mesh_true.node
        x_pred=mesh_pred.node
        #disp_mean=((X_true-x_true)**2).sum(dim=1).sqrt().mean()
        mrse=((x_pred-x_true)**2).sum(dim=1).sqrt()#/disp_mean
        mrse_mean_list.append(mrse.mean().item())
        mrse_max_list.append(mrse.max().item())
        mrse_min_list.append(mrse.min().item())
        #----------------------------------------
        if stress == "VM":
            s_true=mesh_true.element_data['VM']
            s_pred=mesh_pred.element_data['VM']
        elif stress == "MP":
            s_true=mesh_true.element_data['S']
            s_true=cal_max_abs_principal_stress(s_true.view(-1,3,3))
            s_pred=mesh_pred.element_data['S']
            try:
                s_pred=cal_max_abs_principal_stress(s_pred.view(-1,3,3))
            except:
                s_pred.fill_(0)
        elif stress == "S":
            s_true=mesh_true.element_data['S']
            s_pred=mesh_pred.element_data['S']
            #print(s_true.shape, s_pred.shape)
        s_true_abs_mean=s_true.abs().mean().item()
        #s_true_abs_min=s_true.abs().min().item()
        s_true_abs_max=s_true.abs().max().item()
        s_pred_abs_max=s_pred.abs().max().item()
        PE=(s_pred-s_true).abs()/s_true_abs_mean
        MAPE=PE.mean().item()
        MAPE_list.append(MAPE)
        APE=abs(s_true_abs_max-s_pred_abs_max)/s_true_abs_mean
        #APE=PE.max().item()
        APE_list.append(APE)
    #----------------------------------------
    mrse_mean_list=np.array(mrse_mean_list)
    mrse_max_list=np.array(mrse_max_list)
    mrse_min_list=np.array(mrse_min_list)
    #----------------------------------------
    MAPE_list=np.array(MAPE_list)
    #if np.isnan(MAPE_list).sum()>0:
    #    print("np.isnan(MAPE_list).sum()>0: set nan to 1")
    MAPE_list[np.isnan(MAPE_list)==True]=1
    #----------------------------------------
    APE_list=np.array(APE_list)
    #if np.isnan(APE_list).sum()>0:
    #    print("np.isnan(APE_list).sum()>0: set nan to 1")
    APE_list[np.isnan(APE_list)==True]=1
    #if np.sum(APE_list>1)>0:
    #    print("np.sum(APE_list>1)>0")
    return mrse_mean_list, mrse_max_list, mrse_min_list, MAPE_list, APE_list
#%%
def get_time_cost(filelist_true, filelist_pred):
    time_true=[]
    time_pred=[]
    for file_true, file_pred in zip(filelist_true, filelist_pred):
        mesh_true=PolyhedronMesh()
        mesh_true.load_from_torch(file_true+'.pt')
        mesh_pred=PolyhedronMesh()
        mesh_pred.load_from_torch(file_pred+'.pt')
        try:
            time_pred.append(mesh_pred.mesh_data['time'][-1])
            time_true.append(mesh_true.mesh_data['time'][-1])
        except:
            #print('get_time_cost error:', file_true, file_pred )
            pass
    if len(time_pred) > 0:
        time_true=np.array(time_true)
        time_pred=np.array(time_pred)
        time_cost=time_pred/(1e-10+time_true)
    else:
        time_cost=np.zeros(len(filelist_true))
    return time_cost
#%%
def get_filelist(net, folder_data, folder_result, test_or_val, refine, iter_threshold=None):
    (filelist_train_p0,
     filelist_train,
     filelist_val,
     filelist_test)=train_val_test_split(folder_data)

    if test_or_val == 'test':
        filelist_true=filelist_test
        folder_pred=folder_result+net+"/test/"

    elif test_or_val == 'val':
        filelist_true=filelist_val
        folder_pred=folder_result+net+"/val/"
    else:
        filelist_true=filelist_train
        folder_pred=folder_result+net+"/train/"
    filelist_pred=[]
    for n in range(0, len(filelist_true)):
        name_true=filelist_true[n][1]
        name_true=name_true.split("/")[-1]
        if refine == False:
            name_pred=folder_pred+"pred_"+name_true
        else:
            name_pred=folder_pred+"pred_"+name_true+"_refine_R1"
        filelist_pred.append(name_pred)
        #print(name_pred)

    idlist=[]
    for n in range(0, len(filelist_pred)):
        a=os.path.isfile(filelist_pred[n]+'.pt')
        if a == True:
            if iter_threshold is None:
                idlist.append(n)
            else:
                mesh_pred=PolyhedronMesh()
                mesh_pred.load_from_torch(filelist_pred[n]+'.pt')
                if len(mesh_pred.mesh_data['time']) <= iter_threshold:
                    idlist.append(n)

    error_counter=len(filelist_pred)-len(idlist)
    filelist_true=np.array(filelist_true)
    filelist_true=filelist_true[idlist]
    filelist_true=filelist_true[:,1]
    filelist_pred=np.array(filelist_pred)
    filelist_pred=filelist_pred[idlist]
    return filelist_true, filelist_pred, error_counter
#%%
def get_table(net, folder_data, folder_result, test_or_val='test', refine=False, stress='VM',
              return_error_counter=False, iter_threshold=None):
    mrse_table=[]
    MAPE_table=[]
    APE_table=[]
    time_table=[]
    filelist_true, filelist_pred, error_counter=get_filelist(net, folder_data, folder_result,
                                                             test_or_val, refine, iter_threshold)
    mrse_mean_list, mrse_max_list, mrse_min_list, MAPE_list, APE_list=compare(filelist_true, filelist_pred, stress)
    mrse_mean=np.mean(mrse_mean_list)
    #mrse_mean=np.median(mrse_mean_list)
    #mrse_max=np.max(mrse_max_list)
    mrse_max=np.max(mrse_mean_list)#paper
    #mrse_min=np.min(mrse_min_list)
    mrse_min=np.min(mrse_mean_list)
    mrse=(mrse_mean, mrse_max, mrse_min)
    MAPE_mean=np.mean(MAPE_list)
    #MAPE_mean=np.median(MAPE_list)
    MAPE_max=np.max(MAPE_list)
    MAPE_min=np.min(MAPE_list)
    MAPE=(MAPE_mean, MAPE_max, MAPE_min)
    APE_mean=np.mean(APE_list)
    #APE_mean=np.median(APE_list)
    APE_max=np.max(APE_list)
    APE_min=np.min(APE_list)
    APE=(APE_mean, APE_max, APE_min)
    time_cost=get_time_cost(filelist_true, filelist_pred)
    time_cost_mean=np.mean(time_cost)
    time_cost_max=np.max(time_cost)
    time_cost_min=np.min(time_cost)
    time_cost=(time_cost_mean, time_cost_max, time_cost_min)
    mrse_table.append(mrse)
    MAPE_table.append(MAPE)
    APE_table.append(APE)
    time_table.append(time_cost)
    if return_error_counter == False:
        return mrse_table, MAPE_table, APE_table, time_table
    else:
        return mrse_table, MAPE_table, APE_table, time_table, error_counter
#%%
def get_result(net, folder_data, folder_result, test_or_val='test', stress='VM', refine=False, iter_threshold=None):
    filelist_true, filelist_pred, error=get_filelist(net, folder_data, folder_result,
                                                     test_or_val, refine, iter_threshold)
    mrse_mean, mrse_max, mrse_min, MAPE_list, APE_list=compare(filelist_true, filelist_pred, stress)
    time_cost=get_time_cost(filelist_true, filelist_pred)
    return mrse_mean, mrse_max, MAPE_list, APE_list, time_cost, filelist_true, filelist_pred
#%%
def get_data_frame(net, name, folder_data, folder_result, test_or_val, stress, refine, iter_threshold=None):
    mrse, MAPE, APE, t, error=get_table(net, folder_data, folder_result, test_or_val=test_or_val,
                                        stress=stress, refine=refine, return_error_counter=True,
                                        iter_threshold=iter_threshold)
    frame=[]
    frame.append('{a:.4f}'.format(a=mrse[0][0]))
    frame.append('{a:.4f}'.format(a=mrse[0][1]))
    frame.append('{a:.4f}%'.format(a=100*MAPE[0][0]))
    if MAPE[0][0] > 1:
        frame[2]='>100%'
    frame.append('{a:.4f}%'.format(a=100*MAPE[0][1]))
    if MAPE[0][1] > 1:
        frame[3]='>100%'
    frame.append('{a:.4f}%'.format(a=100*APE[0][0]))
    if APE[0][0] > 1:
        frame[4]='>100%'
    frame.append('{a:.4f}%'.format(a=100*APE[0][1]))
    if APE[0][1] > 1:
        frame[5]='>100%'
    if refine == True:
        frame.append('{a:.4f}%'.format(a=100*t[0][0]))
        frame.append('{a:.4f}%'.format(a=100*t[0][1]))
        frame.append(error)
        columns=['MRSE_avg', 'MRSE_max', 'MAPE_avg', 'MAPE_max', 'APE_avg', 'APE_max',
                 'Time_avg', 'Time_max', 'error']
    else:
        columns=['MRSE_avg', 'MRSE_max', 'MAPE_avg', 'MAPE_max', 'APE_avg', 'APE_max']
    frame=pd.DataFrame([frame], columns=columns, index=[name])
    return frame