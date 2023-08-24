import sys
sys.path.append("./mesh")
import numpy as np
import torch
import glob
import pandas as pd
from PolyhedronMesh import PolyhedronMesh
#%%
pd.set_option("display.max_columns", None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 1000)
#%%
def get_error(filename, save=False):
    Mesh_mat=PolyhedronMesh()
    Mesh_mat.load_from_torch(filename+".pt")
    loss=Mesh_mat.mesh_data['loss'].numpy().reshape(-1)
    loss=loss[-1]
    time=Mesh_mat.mesh_data['time'].numpy().reshape(-1)
    time=time[-1]
    Mat_node_true=Mesh_mat.node_data['Mat_true']
    Mat_node_pred=Mesh_mat.node_data['Mat_pred']
    error_node=(Mat_node_pred-Mat_node_true).abs()/Mat_node_true.mean(dim=0, keepdim=True)
    Mat_element_true=Mesh_mat.element_data['Mat_true']
    Mat_element_pred= Mesh_mat.element_data['Mat_pred']
    error_element=(Mat_element_pred-Mat_element_true).abs()/Mat_element_true.mean(dim=0, keepdim=True)
    print('error_element', error_element.shape)
    if save==True:
        Mesh_mat.node_data['Error']=error_node
        Mesh_mat.element_data['Error']=error_element
        Mesh_mat.save_by_vtk(filename+".vtk")
        Mesh_mat.save_by_torch(filename+".pt")
    error_mean=error_element.mean(dim=0).numpy()
    error_mean=error_mean[0:-1]
    error_max=error_element.max(dim=0)[0].numpy()
    error_max=error_max[0:-1]
    return error_mean, error_max, loss, time
#%%
path='./result/inverse/mat_distribution2_mean_shape/'
#%% ['24', '150', '168', '171', '174', '192', '318']
id='171'
filename_base='p0_'+id+'_solid_mat_distribution2_p18_mat_net_'
#%%
table_net3={'layers':[], 'units':[], 'm0':[], 'm1':[], 'm2':[],'m3':[], 'm4':[], 'loss':[], 't':[]}
for i, m in enumerate(['2', '4', '6']):
    for j, n in enumerate(['128', '256', '512']):
        table_net3['layers'].append(i*2+4)
        table_net3['units'].append(n)
        net="Net3(3,"+n+","+m+",1,1,1,5)"
        filename=path+filename_base+net
        #print(filename)
        error_mean, error_max, loss, t=get_error(filename)
        error_mean=error_mean*100
        error_max=error_max*100
        table_net3['m0'].append('{a:.4f}% ({b:.4f}%)'.format(a=error_mean[0], b=error_max[0]))
        table_net3['m1'].append('{a:.4f}% ({b:.4f}%)'.format(a=error_mean[1], b=error_max[1]))
        table_net3['m2'].append('{a:.4f}% ({b:.4f}%)'.format(a=error_mean[2], b=error_max[2]))
        table_net3['m3'].append('{a:.4f}% ({b:.4f}%)'.format(a=error_mean[3], b=error_max[3]))
        table_net3['m4'].append('{a:.4f}% ({b:.4f}%)'.format(a=error_mean[4], b=error_max[4]))
        table_net3['loss'].append(loss)
        table_net3['t'].append(t)
table_net3=pd.DataFrame(table_net3)
print(table_net3)
#table_net3.to_csv(path+"table"+id+"_ex_vivo_inverse_mat_net3.csv", index=False)
#%%
table_net0={'layers':[], 'units':[], 'm0':[], 'm1':[], 'm2':[],'m3':[], 'm4':[], 'loss':[], 't':[]}
for i, m in enumerate(['2', '4', '6']):
    for j, n in enumerate(['128', '256', '512']):
        table_net0['layers'].append(i*2+4)
        table_net0['units'].append(n)
        net="Net0(3,"+n+","+m+",1,1,1,5)"
        filename=path+filename_base+net
        #print(filename)
        error_mean, error_max, loss, t=get_error(filename)
        error_mean=error_mean*100
        error_max=error_max*100
        table_net0['m0'].append('{a:.4f}% ({b:.4f}%)'.format(a=error_mean[0], b=error_max[0]))
        table_net0['m1'].append('{a:.4f}% ({b:.4f}%)'.format(a=error_mean[1], b=error_max[1]))
        table_net0['m2'].append('{a:.4f}% ({b:.4f}%)'.format(a=error_mean[2], b=error_max[2]))
        table_net0['m3'].append('{a:.4f}% ({b:.4f}%)'.format(a=error_mean[3], b=error_max[3]))
        table_net0['m4'].append('{a:.4f}% ({b:.4f}%)'.format(a=error_mean[4], b=error_max[4]))
        table_net0['loss'].append(loss)
        table_net0['t'].append(t)
table_net0=pd.DataFrame(table_net0)
print(table_net0)
#table_net0.to_csv(path+"table"+id+"_ex_vivo_inverse_mat_net0.csv", index=False)
#%%
table_net_none={'m0':[], 'm1':[], 'm2':[],'m3':[], 'm4':[], 'loss':[], 't':[]}
error_mean, error_max, loss, t=get_error(path+filename_base+'none')
error_mean=error_mean*100
error_max=error_max*100
table_net_none['m0'].append('{a:.4f}% ({b:.4f}%)'.format(a=error_mean[0], b=error_max[0]))
table_net_none['m1'].append('{a:.4f}% ({b:.4f}%)'.format(a=error_mean[1], b=error_max[1]))
table_net_none['m2'].append('{a:.4f}% ({b:.4f}%)'.format(a=error_mean[2], b=error_max[2]))
table_net_none['m3'].append('{a:.4f}% ({b:.4f}%)'.format(a=error_mean[3], b=error_max[3]))
table_net_none['m4'].append('{a:.4f}% ({b:.4f})%'.format(a=error_mean[4], b=error_max[4]))
table_net_none['loss'].append(loss)
table_net_none['t'].append(t)
table_net_none=pd.DataFrame(table_net_none)
print(table_net_none)
#table_net_none.to_csv(path+"table"+id+"_ex_vivo_inverse_mat_net_none.csv", index=False)
#%%
net='Net3(3,512,6,1,1,1,5)'
Mesh_mat=PolyhedronMesh()
Mesh_mat.load_from_torch(path+filename_base+net+".pt")
error=Mesh_mat.mesh_data['error'][:,0,0:5].mean(dim=1).numpy()
time=Mesh_mat.mesh_data['time'].numpy()
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(time/60, error)
ax.set_ylim(0, 0.1)
ax.set_yticks(np.arange(0,0.11,0.01))
ax.set_xlim(0, 240)
ax.set_xticks(np.arange(0,250,10))
ax.set_ylabel('error', fontsize=12)
ax.set_xlabel('time in minute', fontsize=12)
ax.grid(True)

