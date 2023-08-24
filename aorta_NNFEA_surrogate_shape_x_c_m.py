import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import sys
sys.path.append("./c3d8")
sys.path.append("./mesh")
import numpy as np
from IPython import display
import matplotlib.pyplot as plt
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark=True
from torch.optim import Adamax
from QuadMesh import QuadMesh
from PolyhedronMesh import PolyhedronMesh
from aorta_mesh import get_solid_mesh_cfg
import time
from copy import deepcopy
from NNFEA_net_x_c_m import Encoder3, Net1, NetXCM1, NetXCM1A
from train_val_test_split_x_c_m import train_val_test_split
#%%
import argparse
parser = argparse.ArgumentParser(description='Input Parameters:')
parser.add_argument('--cuda', default=0, type=int)
parser.add_argument('--dtype', default="float32", type=str)
parser.add_argument('--shell_mesh', default='./data/343c1.5_125mat/bav17_AortaModel_P0_best', type=str)
parser.add_argument('--mesh_tube', default='./data/aorta_tube_solid_1layers', type=str)
parser.add_argument('--folder_data', default='./data/343c1.5_125mat/', type=str)
parser.add_argument('--folder_result', default='./result/forward/', type=str)
parser.add_argument('--max_epochs', default=5000, type=int)
parser.add_argument('--lr_decay_per_epochs', default=100, type=int)
parser.add_argument('--lr_decay', default=None, type=float)
parser.add_argument('--lr_init', default=1e-3, type=float)
parser.add_argument('--lr_min', default=1e-5, type=float)
parser.add_argument('--num_workers', default=2, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--normalized_mat_range', default=[0,1], type=list)#[0,1] or [-1,1]
parser.add_argument('--net', default="NetXCM1A", type=str)
parser.add_argument('--encoder_net', default="Encoder3('BaseNet0',3,128,2,1,1,1,5)", type=str)
parser.add_argument('--decoder_net', default="Net1('BaseNet5b',3,10,256,4,1,1,1,3,'softplus')", type=str)
parser.add_argument('--n_models', default=1, type=int)
arg = parser.parse_args()
#%%
if arg.lr_decay is None:
    arg.lr_decay=np.exp(np.log(arg.lr_min/arg.lr_init)/(arg.max_epochs//arg.lr_decay_per_epochs-1))
print(arg)
#%%
device=torch.device("cuda:"+str(arg.cuda))
#device=torch.device("cpu")
if arg.dtype == "float64":
    dtype=torch.float64
elif arg.dtype == "float32":
    dtype=torch.float32
else:
    raise ValueError("unkown dtype:"+arg.dtype)
#%%
filename_shell=arg.shell_mesh+".pt"
n_layers=1
if '4layer' in arg.mesh_tube:
    n_layers=4
(boundary0, boundary1, Element_surface_pressure, Element_surface_free)=get_solid_mesh_cfg(filename_shell, n_layers)
#%%
mesh_tube=PolyhedronMesh()
mesh_tube.load_from_torch(arg.mesh_tube+".pt")
#%%
NodeTube=mesh_tube.node.to(dtype).to(device)
Element=mesh_tube.element.to(device)
Element_surface_pressure=Element_surface_pressure.to(device)
mask=torch.ones_like(NodeTube)
mask[boundary0]=0
mask[boundary1]=0
#%%
free_node=np.arange(0, NodeTube.shape[0], 1)
free_node=np.setdiff1d(free_node, boundary0.view(-1).numpy())
free_node=np.setdiff1d(free_node, boundary1.view(-1).numpy())
free_node=torch.tensor(free_node, dtype=torch.int64, device=device)
#%%
from torch.utils.data import Dataset, DataLoader
class MeshDataset(Dataset):
    def __init__(self, filelist, preload=False):
        self.filelist=filelist
        self.preload=preload
        if preload == True:
            X_list=[];x_list=[]; m_list=[]
            for n in range(0, len(filelist)):
                name_p0=self.filelist[n][0]
                name_px=self.filelist[n][1]
                m=self.filelist[n][2]
                mesh_p0=PolyhedronMesh()
                mesh_p0.load_from_torch(name_p0+'.pt')
                mesh_px=PolyhedronMesh()
                mesh_px.load_from_torch(name_px+'.pt')
                X_list.append(mesh_p0.node)
                x_list.append(mesh_px.node)
                m_list.append(m)
            self.X_list=X_list
            self.x_list=x_list
            self.m_list=m_list
    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        if self.preload == True:
            X=self.X_list[idx]
            x=self.x_list[idx]
            m=self.m_list[idx]
        else:
            name_p0=self.filelist[idx][0]
            name_px=self.filelist[idx][1]
            m=self.filelist[idx][2]
            mesh_p0=PolyhedronMesh()
            mesh_p0.load_from_torch(name_p0+'.pt')
            mesh_px=PolyhedronMesh()
            mesh_px.load_from_torch(name_px+'.pt')
            X=mesh_p0.node
            x=mesh_px.node
        return  X, x, m
#%%
(filelist_train_p0,
 filelist_train,
 filelist_val,
 filelist_test)=train_val_test_split(arg.folder_data)
#-----------------------------
dataset_train=MeshDataset(filelist_train)
loader_train=DataLoader(dataset_train, batch_size=arg.batch_size, num_workers=arg.num_workers,
                        shuffle=True, pin_memory=True)
#---------
dataset_val=MeshDataset(filelist_val, False)
dataset_test=MeshDataset(filelist_test, False)
#---------
dataset_train_s=MeshDataset(filelist_train[::100])
dataset_val_s=MeshDataset(filelist_val[::10])
dataset_test_s=MeshDataset(filelist_test[::10])
#%%
print("train", len(filelist_train), "val", len(filelist_val), "test", len(filelist_test))
#%%
data=[]
for filename_p0 in filelist_train_p0:
    mesh_p0=PolyhedronMesh()
    mesh_p0.load_from_torch(filename_p0+'.pt')
    data.append(mesh_p0.node.view(1, -1))
del mesh_p0
data=torch.cat(data, dim=0)
MeanShape=data.mean(dim=0).reshape(-1,3)
MeanShape=MeanShape.to(dtype).to(device)
#%%
import os
net_name=arg.net+'_'+arg.encoder_net+'_'+arg.decoder_net+'_'+str(arg.n_models)
folder_result=arg.folder_result+net_name+"/"
filename_save=folder_result+net_name+".pt"
folder_result_train=folder_result+'train/'
if os.path.exists(folder_result_train) == False:
    os.makedirs(folder_result_train)
folder_result_val=folder_result+'val/'
if os.path.exists(folder_result_val) == False:
    os.makedirs(folder_result_val)
folder_result_test=folder_result+'test/'
if os.path.exists(folder_result_test) == False:
    os.makedirs(folder_result_test)
#%% NN to solve for the displacement field
NetU=eval(arg.net+'('+str(arg.n_models)+',"'+arg.encoder_net+'","'+arg.decoder_net+'")')
#%% load and test
if arg.max_epochs==0:
    print('load pretrained model')
    state=torch.load(filename_save, map_location='cpu')
    model_state_best=state["model_state_best"]
    model_state_last=state["model_state_last"]
    NetU.load_state_dict(model_state_last)
#%%
NetU.to(dtype).to(device)
#%%
n_parameters=0
for p in NetU.parameters():
    n_parameters+=torch.numel(p)
print("the number of parameters is", n_parameters)
print("n_parameters/n_samples", n_parameters/(len(filelist_train)*1e4))
#%%
from FEModel_C3D8 import cal_F_tensor_8i, cal_F_tensor_1i, cal_attribute_on_node
from von_mises_stress import cal_von_mises_stress
from AortaFEModel_C3D8_SRI import cal_element_orientation
from Mat_GOH_SRI import cal_cauchy_stress
#-----------------------------------------------
mat_all=torch.load(arg.folder_data+'125mat.pt')['mat']
mat_all=mat_all[:,0:5]
mat_all[:,4]=np.pi*(mat_all[:,4]/180)
mat_all=torch.tensor(mat_all, dtype=dtype, device=device)
#%%
#raise ValueError()
#%%
#-----------------------------------------------
def normalize_mat(mat):
    a=arg.normalized_mat_range[0]#min
    b=arg.normalized_mat_range[1]#max
    mat_new=torch.zeros_like(mat)
    if len(mat.shape) == 1:
        mat_new[0]=a+(b-a)*mat[0]/120
        mat_new[1]=a+(b-a)*mat[1]/6000
        mat_new[2]=a+(b-a)*mat[2]/60
        mat_new[3]=a+(b-a)*mat[3]*3
        mat_new[4]=a+(b-a)*mat[4]/(np.pi/2)
    else:
        mat_new[:,0]=a+(b-a)*mat[:,0]/120
        mat_new[:,1]=a+(b-a)*mat[:,1]/6000
        mat_new[:,2]=a+(b-a)*mat[:,2]/60
        mat_new[:,3]=a+(b-a)*mat[:,3]*3
        mat_new[:,4]=a+(b-a)*mat[:,4]/(np.pi/2)
    return mat_new
#-----------------------------------------------
def cal_stress_field(Node_x, Element, Node_X, Mat):
    Fd=cal_F_tensor_8i(Node_x, Element, Node_X)
    Fv=cal_F_tensor_1i(Node_x, Element, Node_X)
    Orientation=cal_element_orientation(Node_X, Element)
    Sd, Sv=cal_cauchy_stress(Fd, Fv, Mat, Orientation, create_graph=False, return_W=False)
    S=Sd+Sv
    S_element=S.mean(dim=1)
    VM_element=cal_von_mises_stress(S_element)
    S_node=cal_attribute_on_node(Node_x.shape[0], Element, S_element)
    VM_node=cal_von_mises_stress(S_node)
    return S_element, VM_element, S_node, VM_node
#-----------------------------------------------
def save(u_pred, filename_p0, filename_px, folder, Mat):
    mesh_p0=PolyhedronMesh()
    mesh_p0.load_from_torch(filename_p0+'.pt')
    #-------------------------------------
    Node_X=mesh_p0.node.to(dtype).to(device)
    Node_x=Node_X+u_pred
    Element=mesh_p0.element.to(device)
    #-------------------------------------
    S_element, VM_element, S_node, VM_node=cal_stress_field(Node_x, Element, Node_X, Mat)
    #-------------------------------------
    mesh_px_pred=PolyhedronMesh()
    mesh_px_pred.element=Element.detach().cpu()
    mesh_px_pred.node=Node_x.detach().cpu()
    mesh_px_pred.element_data['S']=S_element.view(-1,9).detach().cpu()
    mesh_px_pred.element_data['VM']=VM_element.view(-1,1).detach().cpu()
    mesh_px_pred.node_data['S']=S_node.view(-1,9).detach().cpu()
    mesh_px_pred.node_data['VM']=VM_node.view(-1,1).detach().cpu()
    filename_save=folder+'pred_'+filename_px.split('/')[-1]
    mesh_px_pred.save_by_torch(filename_save+".pt")
    #mesh_px_pred.save_by_vtk(filename_save+".vtk")
    #print("saved:", filename_save)
#%%
@torch.no_grad()
def test(dataset, save_file=False, folder=None):
    NetU.eval()
    mrse_mean=[]
    mrse_max=[]
    for n in range(0, len(dataset)):
        X, x_true, m = dataset[n]
        X=X.to(dtype).to(device)
        x_true=x_true.to(dtype).to(device)
        mat=mat_all[m].to(dtype).to(device)
        u_pred=NetU(X, MeanShape, normalize_mat(mat))
        u_pred[boundary0]=0
        u_pred[boundary1]=0
        x_pred=u_pred+X
        mrse=((x_pred-x_true)**2).sum(dim=1).sqrt()
        mrse_mean.append(mrse.mean().item())
        mrse_max.append(mrse.max().item())
        if save_file==True:
            filelist=dataset.filelist
            filename_p0=filelist[n][0]
            filename_px=filelist[n][1]
            save(u_pred, filename_p0, filename_px, folder, mat.view(1,-1))
    mrse_mean=np.mean(mrse_mean)
    mrse_max=np.max(mrse_max)
    return mrse_mean, mrse_max
#%%
mrse_list_train=[]
mrse_list_val=[]
mrse_list_test=[]
#%%
def update_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr
#%%
lr=arg.lr_init
optimizer=Adamax(NetU.parameters(), lr=lr)
#%%
if arg.max_epochs > 0:
    model_state_best=deepcopy(NetU.state_dict())
#%%
for epoch in range(0, arg.max_epochs):
    t0=time.time()
    NetU.train()
    for batch_id, (batch_X, batch_x_true, batch_m) in enumerate(loader_train):
        def closure():
            loss=0
            batch_size=len(batch_X)
            for n in range(0, batch_size):
                X=batch_X[n];  x_true=batch_x_true[n]; m=batch_m[n]
                X=X.to(dtype).to(device)
                x_true=x_true.to(dtype).to(device)
                mat=mat_all[m].to(dtype).to(device)
                u_pred=NetU(X, MeanShape, normalize_mat(mat))
                u_true=x_true-X
                loss=loss+((u_pred-u_true)**2).mean()
            if batch_size > 1:
                loss=loss/float(batch_size)
            if loss.requires_grad==True:
                optimizer.zero_grad()
                loss.backward()
            return loss
        optimizer.step(closure)
    #-------------------
    if (epoch+1)%arg.lr_decay_per_epochs == 0:
        lr=lr*arg.lr_decay
        lr=max(lr, arg.lr_min)
        update_lr(optimizer, lr)
    #-------------------
    t1=time.time()

    mrse_train=test(dataset_train_s)
    mrse_val=test(dataset_val_s)
    mrse_test=test(dataset_test_s)

    T=arg.lr_decay_per_epochs//5
    if len(mrse_list_val) < T:
        mrse_val_best=mrse_val[0]
    else:#moving average over T epochs
        temp=np.convolve(np.array(mrse_list_val)[:,0], np.ones(T)/T, mode='valid')
        mrse_val_best=temp.min()
    if mrse_val[0] < mrse_val_best:
        model_state_best=deepcopy(NetU.state_dict())
        print('record the current best model')

    mrse_list_train.append(mrse_train)
    mrse_list_val.append(mrse_val)
    mrse_list_test.append(mrse_test)

    t2=time.time()
    print("epoch", epoch, "time", t2-t0, t2-t1)
    print("train: mrse", *mrse_train)
    print("val:   mrse", *mrse_val)
    print("test:  mrse", *mrse_test)
    if (epoch+1)%100==0:
        display.clear_output(wait=False)
        fig, ax=plt.subplots(2,1, sharex=True)
        ax[0].plot(np.array(mrse_list_train)[:,0], 'b', linewidth=1)
        ax[0].plot(np.array(mrse_list_val)[:,0],   'g', linewidth=1)
        ax[1].plot(np.array(mrse_list_test)[:,0],  'r', linewidth=1)
        for i in range(0, 2):
            ax[i].set_ylim(0, 0.1)
            ax[i].set_yticks(np.arange(0, 0.11, 0.01))
            ax[i].grid(True)
        display.display(fig)
        plt.close(fig)
#%%
model_state_last=deepcopy(NetU.state_dict())
#%% use the last because the val set is too small
#NetU.load_state_dict(model_state_best)
#%% test
#mrse_train=test(dataset_train, True, folder_result_train)
mrse_val=test(dataset_val, True, folder_result_val)
mrse_test=test(dataset_test, True, folder_result_test)
print("val: mrse", mrse_val)
print("test: mrse", mrse_test)
#%%
if arg.max_epochs > 0:
    torch.save({"arg":arg,
                "model_state_best":model_state_best,
                "model_state_last":model_state_last,
                "MeanShape":MeanShape,
                "mrse_list_train":mrse_list_train,
                "mrse_list_val":mrse_list_val,
                "mrse_list_test":mrse_list_test
                },
               filename_save)
    print("saved:", filename_save)
#%%

