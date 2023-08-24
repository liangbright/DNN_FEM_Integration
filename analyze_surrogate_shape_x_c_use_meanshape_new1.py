import sys
sys.path.append("./c3d8")
sys.path.append("./mesh")
#%%
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analyze_surrogate_shape_x_c_functions import get_data_frame, get_result #, get_table
#%%
pd.set_option("display.max_columns", None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 1000)
#%%
folder_data='./data/343c1.5_fast/'
folder_result="./result/forward/"
stress='VM'
loss1_threshold='R1'
#%%
Best_list=["Encoder3('BaseNet0',3,128,2,1,1,1,3)_Net1('BaseNet5b',3,3,512,4,1,1,1,3,'softplus')_float32",
           'TransEncoder4(2,256,2,16)_TransDecoder4()_float32',
           "UNet(512)_TransDecoder4()_float32",
           "MeshGraphNet(3,8,5,128,3,0.1)_float32",
           "Linear_encoder(30000,3,0.1)_MLP1b(3,512,4,30000)_float32",
           ]
best_result_val=[]
best_result_test=[]
best_result_test_r=[]
mrse_train=[]
mrse_val=[]
for name in Best_list:
    try:
        folder_net=folder_result+name+"/0.5/matMean/"+name+".pt"
        data=torch.load(folder_net, map_location='cpu')
    except:
        folder_net=folder_result+name+"/0.5/matMean/"+name.replace('_float32', '')+".pt"
        data=torch.load(folder_net, map_location='cpu')
    print(name)
    #print(data['arg'])

    mrse_train.append(data['mrse_train'])
    mrse_val.append(data['mrse_val'])

    result_val=get_data_frame(name, name.replace('_float32', ''), folder_data, folder_result,
                              test_or_val='val', stress=stress, refine=False)
    result_test=get_data_frame(name, name.replace('_float32', ''), folder_data, folder_result,
                               test_or_val='test', stress=stress, refine=False)
    result_test_r=get_data_frame(name, name.replace('_float32', ''), folder_data, folder_result,
                               test_or_val='test', stress=stress, refine=True)
    best_result_val.append(result_val)
    best_result_test.append(result_test)
    best_result_test_r.append(result_test_r)
best_result_val=pd.concat(best_result_val)
best_result_test=pd.concat(best_result_test)
best_result_test_r=pd.concat(best_result_test_r)
#%%
idx_best=np.argmin(best_result_val['MRSE_avg'].values)
print("best val", best_result_val.iloc[idx_best])
print("best test", best_result_test.iloc[idx_best])
print("best test_r", best_result_test_r.iloc[idx_best])
#%%
best_result_test.to_csv(folder_result+"NN_result_test.csv")
best_result_test_r.to_csv(folder_result+"NN_result_test_R.csv")
#%%
fig, ax = plt.subplots(2,2,constrained_layout=True, sharex=True, sharey=True)
ax[0,0].plot(np.array(mrse_val[0])[:,0], color='r', label='W-Net'); ax[0,0].set_ylim(0, 1)
#ax[0,0].plot(np.array(mrse_val[0])[:,0], color='m')
ax[0,1].plot(np.array(mrse_val[1])[:,0], color='g', label='TransNet'); ax[0,1].set_ylim(0, 1)
ax[1,0].plot(np.array(mrse_val[2])[:,0], color='b', label='U-Net'); ax[1,0].set_ylim(0, 1)
ax[1,1].plot(np.array(mrse_val[3])[:,0], color='c', label='MeshGraphNet'); ax[1,1].set_ylim(0, 1)
ax[0,0].legend(loc='center')
ax[0,1].legend(loc='center')
ax[1,0].legend(loc='center')
ax[1,1].legend(loc='center')
#%%
for k in range(0, 4):
    mrse_train_k=np.array(mrse_train[k])[:,0]
    print(abs(mrse_train_k[-1]-mrse_train_k[-2]))
for k in range(0, 4):
    mrse_val_k=np.array(mrse_val[k])[:,0]
    print(abs(mrse_val_k[-1]-mrse_val_k[-2]))
#%%
result=get_result("Encoder3('BaseNet0',3,128,2,1,1,1,3)_Net1('BaseNet5b',3,3,512,4,1,1,1,3,'softplus')_float32",
                  folder_data, folder_result, train_percent=0.5, test_or_val='test', stress='VM', refine=False)
mrse_mean, mrse_max, MAPE_list, APE_list, time_cost, filelist_true, filelist_pred = result
#%%
import numpy as np
print(np.sum(MAPE_list>0.1)/len(MAPE_list))
print(np.sum(APE_list>0.1)/len(APE_list))

print(np.sum(MAPE_list>0.05)/len(MAPE_list))
print(np.sum(APE_list>0.05)/len(APE_list))

#%%
from PolyhedronMesh import PolyhedronMesh
from train_val_test_split_x_c_new1 import train_val_test_split

(filelist_train, filelist_val, filelist_test,
 shape_idlist_train, shape_idlist_val, shape_idlist_test)=train_val_test_split(folder_data, 0.5)
#%%
def load_mesh(filename_px, folder):
    mesh_px=PolyhedronMesh()
    mesh_px.load_from_torch(filename_px)
    mesh_p0=PolyhedronMesh()
    filename_p0=filename_px.replace('i90', 'i0')
    mesh_p0.load_from_torch(filename_p0)
    X=mesh_p0.node
    x=mesh_px.node
    return X, x
#%%
def load_all(filelist, folder):
    X_all=[]; x_all=[]
    for filename_px in filelist:
        try:
            X, x=load_mesh(filename_px, folder)
            X_all.append(X)
            x_all.append(x)
        except:
            print("cannot load", filename_px)
    return X_all, x_all
#%%
X_train, x_train=load_all(filelist_train, folder_data)
X_val, x_val=load_all(filelist_val, folder_data)
X_test, x_test=load_all(filelist_test, folder_data)
#%%
from sklearn.decomposition import PCA
data_train=[]
for X in X_train:
    data_train.append(X.view(1, -1).cpu())
data_train=torch.cat(data_train, dim=0)
data_train=data_train.numpy()
pca=PCA(n_components=10)
pca.fit(data_train)
#%%
data_test=[]
for x in x_test:
    data_test.append(x.view(1, -1).cpu())
data_test=torch.cat(data_test, dim=0)
data_test=data_test.numpy()
data_test_rec=pca.inverse_transform(pca.transform(data_test))
#%%
data_test=data_test.reshape(-1,10000,3)
data_test_rec=data_test_rec.reshape(-1,10000,3)
rec_error_test=np.sqrt(((data_test_rec-data_test)**2).sum(axis=2)).mean(axis=1)
#%%
from sklearn.metrics import roc_auc_score
Best_list=["Encoder3('BaseNet0',3,128,2,1,1,1,3)_Net1('BaseNet5b',3,3,512,4,1,1,1,3,'softplus')_float32",
           'TransEncoder4(2,256,2,16)_TransDecoder4()_float32',
           "UNet(512)_TransDecoder4()_float32",
           "MeshGraphNet(3,8,5,128,3,0.1)_float32",
           "Linear_encoder(30000,3,0.1)_MLP1b(3,512,4,30000)_float32",
           ]
auc_list_rec10=[]
auc_list_rec05=[]
auc_list_rec01=[]
for name in Best_list:

    result_test=get_result(name, folder_data, folder_result, 0.5,
                           test_or_val='test',  stress=stress, refine=False)
    try:
        auc10 = roc_auc_score(result_test[3]>0.10, rec_error_test)
        auc05 = roc_auc_score(result_test[3]>0.05, rec_error_test)
        auc01 = roc_auc_score(result_test[3]>0.01, rec_error_test)
    except:
        auc10=0.5
        auc05=0.5
        auc01=0.5
    auc_list_rec10.append(auc10)
    auc_list_rec05.append(auc05)
    auc_list_rec01.append(auc01)
    fig, ax = plt.subplots()
    #ax.hist(result_test[3], bins=100)
    ax.plot(rec_error_test, result_test[3], '.')
    #ax.set_ylim(0,1)
    ax.set_title(name)
    ax.set_xlabel(str(auc10)+' '+str(auc05)+' '+str(auc01))
#%%
from AortaFEModel_C3D8_SRI import AortaFEModel
from Mat_GOH_SRI import cal_1pk_stress, cal_cauchy_stress
from aorta_mesh import get_solid_mesh_cfg
device=torch.device("cuda:0")
dtype=torch.float64
#%%
filename_shell='./data/bav17_AortaModel_P0_best.pt'
(boundary0, boundary1, Element_surface_pressure, Element_surface_free)=get_solid_mesh_cfg(filename_shell)
Mat=torch.load('./data/125mat.pt')['mean_mat_str']
Mat=[float(m) for m in Mat.split(",")]
Mat[4]=np.pi*(Mat[4]/180)
Mat=torch.tensor([Mat], dtype=dtype, device=device)
#%%
auc_list_Rmean10=[]
auc_list_Rmean05=[]
auc_list_Rmean01=[]
auc_list_Rmax10=[]
auc_list_Rmax05=[]
auc_list_Rmax01=[]
auc_list_loss10=[]
auc_list_loss05=[]
auc_list_loss01=[]
for name in Best_list:
    result_test=get_result(name, folder_data, folder_result, 0.5,
                           test_or_val='test',  stress=stress, refine=False)
    filelist_test_pred=result_test[-1]
    loss1_list=[]
    Rmean_list=[]
    Rmax_list=[]
    for k in range(0, len(filelist_test)):
        mesh_p0=filelist_test[k].replace("i90", "i0")
        mesh_px=filelist_test_pred[k]
        pressure=18

        Mesh_X=PolyhedronMesh()
        Mesh_X.load_from_torch(mesh_p0)
        Node_X=Mesh_X.node.to(dtype).to(device)
        Element=Mesh_X.element.to(device)

        Mesh_x=PolyhedronMesh()
        Mesh_x.load_from_torch(mesh_px)
        Node_x=Mesh_x.node.to(dtype).to(device)

        aorta_model=AortaFEModel(Node_x, Element, Node_X, boundary0, boundary1, Element_surface_pressure,
                                 Mat, cal_1pk_stress, cal_cauchy_stress, dtype, device, mode='inflation')
        out=aorta_model.cal_energy_and_force(pressure)

        TPE1=out['TPE1']; TPE2=out['TPE2']; SE=out['SE']
        force_int=out['force_int']; force_ext=out['force_ext']
        force_int_of_element=out['force_int_of_element'].detach()
        loss1=((force_int-force_ext)**2).sum(dim=1).sqrt().mean().item()
        loss1_list.append(loss1)
        force_avg=(force_int_of_element**2).sum(dim=2).sqrt().mean()
        force_res=((force_int-force_ext)**2).sum(dim=1).sqrt()
        R=force_res/(force_avg+1e-10)
        Rmean=R.mean().item()
        Rmax=R.max().item()
        Rmax_list.append(Rmax)
        Rmean_list.append(Rmean)
    #-----------
    try:
        auc10 = roc_auc_score(result_test[3]>0.10, Rmean_list)
        auc05 = roc_auc_score(result_test[3]>0.05, Rmean_list)
        auc01 = roc_auc_score(result_test[3]>0.01, Rmean_list)
    except:
        auc10=0.5
        auc05=0.5
        auc01=0.5
    auc_list_Rmean10.append(auc10)
    auc_list_Rmean05.append(auc05)
    auc_list_Rmean01.append(auc01)
    #----------------------------
    try:
        auc10 = roc_auc_score(result_test[3]>0.10, Rmax_list)
        auc05 = roc_auc_score(result_test[3]>0.05, Rmax_list)
        auc01 = roc_auc_score(result_test[3]>0.01, Rmax_list)
    except:
        auc10=0.5
        auc05=0.5
        auc01=0.5
    auc_list_Rmax10.append(auc10)
    auc_list_Rmax05.append(auc05)
    auc_list_Rmax01.append(auc01)
    #----------------------------
    try:
        auc10 = roc_auc_score(result_test[3]>0.10, loss1_list)
        auc05 = roc_auc_score(result_test[3]>0.05, loss1_list)
        auc01 = roc_auc_score(result_test[3]>0.01, loss1_list)
    except:
        auc10=0.5
        auc05=0.5
        auc01=0.5
    auc_list_loss10.append(auc10)
    auc_list_loss05.append(auc05)
    auc_list_loss01.append(auc01)
    #----------------------------
    fig, ax = plt.subplots()
    #ax.hist(result_test[3], bins=100)
    ax.plot(rec_error_test, result_test[3], '.')
    #ax.set_ylim(0,1)
    ax.set_title(name)
    ax.set_xlabel(str(auc10)+' '+str(auc05)+' '+str(auc01))
#%%

df=pd.DataFrame()
df['method']=['rec', "loss1", 'Rmean', 'Rmax']
for k in range(0, len(Best_list)):
    name=Best_list[k][0:3]
    df[name]=[auc_list_rec10[k], auc_list_loss10[k], auc_list_Rmean10[k], auc_list_Rmax10[k]]
print(df)
df.to_csv(folder_result+"ood_detection_0.10.csv", index=False)

df=pd.DataFrame()
df['method']=['rec', "loss1", 'Rmean', 'Rmax']
for k in range(0, len(Best_list)):
    name=Best_list[k][0:3]
    df[name]=[auc_list_rec05[k], auc_list_loss05[k], auc_list_Rmean05[k], auc_list_Rmax05[k]]
print(df)
df.to_csv(folder_result+"ood_detection_0.05.csv", index=False)

df=pd.DataFrame()
df['method']=['rec', "loss1", 'Rmean', 'Rmax']
for k in range(0, len(Best_list)):
    name=Best_list[k][0:3]
    df[name]=[auc_list_rec01[k], auc_list_loss01[k], auc_list_Rmean01[k], auc_list_Rmax01[k]]
print(df)
df.to_csv(folder_result+"ood_detection_0.01.csv", index=False)

