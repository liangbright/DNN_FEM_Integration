import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

import sys
sys.path.append("./c3d8")
sys.path.append("./mesh")
import numpy as np
from IPython import display
import matplotlib.pyplot as plt
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark=True
from QuadMesh import QuadMesh
from PolyhedronMesh import PolyhedronMesh
from aorta_mesh import get_solid_mesh_cfg
import time
import glob
from copy import deepcopy
import logging
from NNFEA_net_x_c import Encoder3, Net1
from NNFEA_net_x_c import Linear_encoder, MLP1b
from NNFEA_net_x_c_trans_new import TransEncoder4, TransDecoder4
from NNFEA_net_x_c_UNet import UNet
from train_val_test_split_x_c_new1 import train_val_test_split
#%%
import argparse
parser = argparse.ArgumentParser(description='Input Parameters:')
parser.add_argument('--cuda', default=0, type=int)
parser.add_argument('--dtype', default="float32", type=str)
parser.add_argument('--shell_mesh', default='./data/bav17_AortaModel_P0_best', type=str)
parser.add_argument('--mesh_tube', default='./data/aorta_tube_solid_1layers', type=str)
parser.add_argument('--folder_data', default='./data/343c1.5_fast/', type=str)
parser.add_argument('--folder_result', default='./result/forward/', type=str)
parser.add_argument('--train_percent', default=0.5, type=float)
parser.add_argument('--max_epochs', default=20000, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--lr_decay_interval', default=100, type=int)
parser.add_argument('--lr_decay', default=None, type=float)
parser.add_argument('--lr_init', default=1e-3, type=float)
parser.add_argument('--lr_min', default=1e-5, type=float)
parser.add_argument('--resume_training', default=0, type=int)
#----W-Net
parser.add_argument('--encoder_net', default="Encoder3('BaseNet0',3,128,2,1,1,1,3)", type=str)
parser.add_argument('--decoder_net', default="Net1('BaseNet5b',3,3,512,4,1,1,1,3,'softplus')", type=str)
#----TransNet
#parser.add_argument('--encoder_net', default="TransEncoder4(2,256,2,16)", type=str)
#parser.add_argument('--decoder_net', default="TransDecoder4()", type=str)
#----UNet
#parser.add_argument('--encoder_net', default="UNet(512)", type=str)
#parser.add_argument('--decoder_net', default="TransDecoder4()", type=str)
#---MLP
#parser.add_argument('--encoder_net', default="Linear_encoder(30000,3,0.1)", type=str)
#parser.add_argument('--decoder_net', default="MLP1b(3,512,4,30000)", type=str)
#--MeshGraphNet is in aorta_NNFEA_surrogate_stress_x_c_meshgraphnet.py
arg = parser.parse_args()
#%%
if arg.lr_decay is None:
    arg.lr_decay=np.exp(np.log(arg.lr_min/arg.lr_init)/(arg.max_epochs//arg.lr_decay_interval-1))
#%%
arg.use_mean_shape=1
print(arg)
#%%
device=torch.device("cpu")
if arg.cuda >= 0:
    device=torch.device("cuda:"+str(arg.cuda))
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
#dtype='float64'
#mesh_tube.load_from_vtk(arg.mesh_tube+".vtk", dtype=dtype)
#mesh_tube.save_by_torch(arg.mesh_tube+".pt")
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
(filelist_train, filelist_val, filelist_test,
 shape_idlist_train, shape_idlist_val, shape_idlist_test)=train_val_test_split(arg.folder_data, arg.train_percent)
#%%
def load_mesh(filename_px, folder, dtype=dtype, device=device):
    with torch.no_grad():
        mesh_px=PolyhedronMesh()
        mesh_px.load_from_torch(filename_px)
        mesh_p0=PolyhedronMesh()
        #filename_p0=folder+mesh_px.mesh_data['arg'].mesh_p0+".pt"
        filename_p0=filename_px.replace('i90', 'i0')
        mesh_p0.load_from_torch(filename_p0)
        X=mesh_p0.node.to(dtype).to(device)
        x=mesh_px.node.to(dtype).to(device)
    return X, x
#%%
def load_all(filelist, folder, dtype, device):
    X_all=[]; x_all=[]
    for filename_px in filelist:
        try:
            X, x=load_mesh(filename_px, folder, dtype, device)
            X_all.append(X)
            x_all.append(x)
        except:
            print("cannot load", filename_px)
    return X_all, x_all
#%%
X_train, x_train=load_all(filelist_train, arg.folder_data, dtype, device)
X_val, x_val=load_all(filelist_val, arg.folder_data, dtype, device)
X_test, x_test=load_all(filelist_test, arg.folder_data, dtype, device)
#%%
data=[]
for X in X_train:
    data.append(X.view(1, -1).cpu())
data=torch.cat(data, dim=0)
#%%
from aorta_mesh import shell_to_solid
aorta_shell=QuadMesh()
#aorta_shell.load_from_vtk(filename_shell, dtype=dtype)
aorta_shell.load_from_torch(filename_shell)
MeanShape_shell=data.mean(dim=0).reshape(-1,3)
MeanShape_shell=MeanShape_shell[0:5000]
if n_layers==1:
    thickness=[2]
elif n_layers==4:
    thickness=[0.5, 0.5, 0.5, 0.5]
MeanShape, MeanShape_element=shell_to_solid(MeanShape_shell, aorta_shell.element, thickness)
MeanShape=MeanShape.to(dtype).to(device)
#%%
#sys.exit()
#%%
Origin=MeanShape.mean(dim=0, keepdim=True)
print('Origin', Origin)
#%%
data_std=(data-data.mean(dim=0,keepdim=True)).std().item()
print('1.5*data_std', 1.5*data_std)
#%% NN to solve for the displacement field
encoder_net=eval(arg.encoder_net).to(dtype).to(device)
decoder_net=eval(arg.decoder_net).to(dtype).to(device)
#%%
n_parameters=0
for p in encoder_net.parameters():
    n_parameters+=torch.numel(p)
for p in decoder_net.parameters():
    n_parameters+=torch.numel(p)
print("the number of parameters is", n_parameters)
print("n_parameters/n_samples", n_parameters/(len(filelist_train)*1e4))
#%%
from FEModel_C3D8 import cal_F_tensor_8i, cal_F_tensor_1i, cal_attribute_on_node
from von_mises_stress import cal_von_mises_stress
from AortaFEModel_C3D8_SRI import cal_element_orientation
from Mat_GOH_SRI import cal_cauchy_stress
#-----------------------------------------------
matMean=torch.load(arg.folder_data+'125mat.pt')['mean_mat_str']
matMean=[float(m) for m in matMean.split(",")]
matMean[4]=np.pi*(matMean[4]/180)
matMean=torch.tensor([matMean], dtype=dtype, device=device)
#-----------------------------------------------
def cal_stress_field(Node_x, Element, Node_X, Mat=matMean):
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
def save(u_pred, filename_px, folder_data, folder_result):
    mesh_px=PolyhedronMesh()
    mesh_px.load_from_torch(filename_px)
    mesh_p0=PolyhedronMesh()
    #filename_p0=folder_data+mesh_px.mesh_data['arg'].mesh_p0+".pt"
    filename_p0=filename_px.replace('i90', 'i0')
    mesh_p0.load_from_torch(filename_p0)
    #-------------------------------------
    Node_X=mesh_p0.node.to(dtype).to(device)
    Node_x=Node_X+u_pred
    Element=mesh_px.element.to(device)
    #-------------------------------------
    S_element, VM_element, S_node, VM_node=cal_stress_field(Node_x, Element, Node_X)
    #-------------------------------------
    mesh_px_pred=PolyhedronMesh()
    mesh_px_pred.element=Element.detach().cpu()
    mesh_px_pred.node=Node_x.detach().cpu()
    mesh_px_pred.element_data['S']=S_element.view(-1,9).detach().cpu()
    mesh_px_pred.element_data['VM']=VM_element.view(-1,1).detach().cpu()
    mesh_px_pred.node_data['S']=S_node.view(-1,9).detach().cpu()
    mesh_px_pred.node_data['VM']=VM_node.view(-1,1).detach().cpu()

    #filename_save=folder_result+mesh_px.mesh_data['arg'].mesh_p0+"_matMean_p18_pred"
    filename_p0=filename_px.split("/")[-1].replace('_matMean_p20_i90.pt', '')
    filename_save=folder_result+filename_p0+"_matMean_p18_pred"

    mesh_px_pred.save_by_torch(filename_save+".pt")
    mesh_px_pred.save_by_vtk(filename_save+".vtk")
    print("saved:", filename_save)
#%%
def test(X_list, x_list):
    encoder_net.eval()
    decoder_net.eval()
    with torch.no_grad():
        mrse_mean=[]
        mrse_max=[]
        for X, x_true in zip(X_list, x_list):
            if arg.use_mean_shape == 0:
                c=encoder_net(X-MeanShape, NodeTube)
                u_pred=decoder_net(NodeTube, c)
            else:
                c=encoder_net(X-MeanShape, MeanShape)
                u_pred=decoder_net(MeanShape, c)
            u_pred[boundary0]=0
            u_pred[boundary1]=0
            x_pred=u_pred+X
            mrse=((x_pred-x_true)**2).sum(dim=1).sqrt()
            mrse_mean.append(mrse.mean().item())
            mrse_max.append(mrse.max().item())
        mrse_mean=np.mean(mrse_mean)
        mrse_max=np.max(mrse_max)
    return mrse_mean, mrse_max
#%%
def test_save(X_list, x_list, filelist, folder_data, folder_result):
    encoder_net.eval()
    decoder_net.eval()
    with torch.no_grad():
        mrse_mean=[]
        mrse_max=[]
        for X, x_true, filename_px in zip(X_list, x_list, filelist):
            if arg.use_mean_shape == 0:
                c=encoder_net(X-MeanShape, NodeTube)
                u_pred=decoder_net(NodeTube, c)
            else:
                c=encoder_net(X-MeanShape, MeanShape)
                u_pred=decoder_net(MeanShape, c)
            u_pred[boundary0]=0
            u_pred[boundary1]=0
            x_pred=u_pred+X
            mrse=((x_pred-x_true)**2).sum(dim=1).sqrt()
            mrse_mean.append(mrse.mean().item())
            mrse_max.append(mrse.max().item())
            save(u_pred, filename_px, folder_data, folder_result)
        mrse_mean=np.mean(mrse_mean)
        mrse_max=np.max(mrse_max)
    return mrse_mean, mrse_max
#%%
def get_loss(X, x_true):
    encoder_net.train()
    decoder_net.train()
    y_true=x_true-X
    if arg.use_mean_shape == 0:
        c=encoder_net(X-MeanShape, NodeTube)
        out=decoder_net(NodeTube, c)
    else:
        c=encoder_net(X-MeanShape, MeanShape)
        out=decoder_net(MeanShape, c)
    if isinstance(out, list):
        loss=0
        y_pred=0*y_true
        for y in out:
            y_pred=y+y_pred
            loss=loss+((y_pred[free_node]-y_true[free_node])**2).mean()
        loss=loss/len(out)
    else:
        y_pred=out
        loss=((y_pred[free_node]-y_true[free_node])**2).mean()
    return loss
#%%
from torch.optim import Adamax
lr=arg.lr_init
#%%
def update_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr
#%%
import os
folder_result=arg.folder_result+arg.encoder_net+'_'+arg.decoder_net+'_'+arg.dtype+"/"+str(arg.train_percent)+'/matMean/'
filename_save=folder_result+arg.encoder_net+'_'+arg.decoder_net+".pt"
folder_result_train=folder_result+'train/'
if os.path.exists(folder_result_train) == False:
    os.makedirs(folder_result_train)
folder_result_val=folder_result+'val/'
if os.path.exists(folder_result_val) == False:
    os.makedirs(folder_result_val)
folder_result_test=folder_result+'test/'
if os.path.exists(folder_result_test) == False:
    os.makedirs(folder_result_test)
#%%
logging.basicConfig(filename=folder_result + "/log.txt", level=logging.INFO,
                    format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.info(str(arg))
#%%
if arg.resume_training == True:
    print("load", filename_save)
    state=torch.load(filename_save, map_location='cpu')
    encoder_net=eval(arg.encoder_net)
    encoder_net.load_state_dict(state['encoder_model_state'])
    encoder_net=encoder_net.to(dtype).to(device)
    decoder_net=eval(arg.decoder_net)
    decoder_net.load_state_dict(state['decoder_model_state'])
    decoder_net=decoder_net.to(dtype).to(device)
#%%
mrse_list_train=[]
mrse_list_val=[]
mrse_list_test=[]
#%%
optimizer_encoder=Adamax(encoder_net.parameters(), lr=lr)
optimizer_decoder=Adamax(decoder_net.parameters(), lr=lr)
#optimizer=Adamax(list(encoder_net.parameters())+list(decoder_net.parameters()), lr=lr)
#%%
model_state_best={'encoder':deepcopy(encoder_net.state_dict()),
                  'decoder':deepcopy(decoder_net.state_dict())}
idxlist=np.arange(0, len(filelist_train))
batch_size=arg.batch_size
for epoch in range(0, arg.max_epochs):
    t0=time.time()
    encoder_net.train()
    decoder_net.train()
    np.random.shuffle(idxlist)
    for n in range(0, len(filelist_train), batch_size):
        if n*batch_size > len(filelist_train):
            break
        X=[]; x_true=[]
        for m in range(0, batch_size):
            id=idxlist[m+n*batch_size]
            X.append(X_train[id])
            x_true.append(x_train[id])
        def closure():
            loss=0
            for m in range(0, batch_size):
                loss=loss+get_loss(X[m], x_true[m])
            loss=loss/batch_size
            if loss.requires_grad==True:
                optimizer_encoder.zero_grad()
                optimizer_decoder.zero_grad()
                #optimizer.zero_grad()
                loss.backward()
            return loss
        optimizer_encoder.step(closure)
        optimizer_decoder.step(closure)
        #optimizer.step(closure)
    #-------------------
    if (epoch+1)%arg.lr_decay_interval == 0:
        lr=lr*arg.lr_decay
        lr=max(lr, arg.lr_min)
        update_lr(optimizer_encoder, lr)
        update_lr(optimizer_decoder, lr)
        #update_lr(optimizer, lr)
    #-------------------

    mrse_train=test(X_train, x_train)
    mrse_val=test(X_val, x_val)
    mrse_test=test(X_test, x_test)
    T=arg.lr_decay_interval//5
    if len(mrse_list_val) < T:
        mrse_val_best=mrse_val[0]
    else:#moving average over T epochs
        temp=np.convolve(np.array(mrse_list_val)[:,0], np.ones(T)/T, mode='valid')
        mrse_val_best=temp.min()
    if mrse_val[0] < mrse_val_best:
        model_state_best['encoder']=deepcopy(encoder_net.state_dict())
        model_state_best['decoder']=deepcopy(decoder_net.state_dict())
        print('record the current best model')
        if mrse_val[0] < 0.02:
            torch.save({"arg":arg,
                        "encoder_model_state":encoder_net.state_dict(),
                        "decoder_model_state":decoder_net.state_dict(),
                        "mrse_train":mrse_list_train,
                        "mrse_val":mrse_list_val,
                        "mrse_test":mrse_list_test},
                       filename_save)
            print("saved:", filename_save)

    mrse_list_train.append(mrse_train)
    mrse_list_val.append(mrse_val)
    mrse_list_test.append(mrse_test)

    t1=time.time()
    #print("epoch", epoch, "time", t1-t0, "loss1", loss1)
    #print("epoch", epoch, "time", t1-t0)
    #print("train: mrse", *mrse_train)
    #print("val:   mrse", *mrse_val)
    #print("test:  mrse", *mrse_test)

    logging.info('epoch: %f, time: %f' % (epoch, t1-t0))
    logging.info('train: %f, %f' % (mrse_train[0], mrse_train[1]))
    logging.info('val  : %f, %f' % (mrse_val[0], mrse_val[1]))
    logging.info('test : %f, %f' % (mrse_test[0], mrse_test[1]))


    if (epoch+1)%100==0:
        display.clear_output(wait=False)
        fig, ax=plt.subplots(2,1, sharex=True)
        ax[0].plot(np.array(mrse_list_train)[:,0], 'b', linewidth=1)
        ax[0].plot(np.array(mrse_list_val)[:,0], 'g', linewidth=1)
        ax[1].plot(np.array(mrse_list_test)[:,0], 'r', linewidth=1)
        for i in range(0, 2):
            ax[i].set_ylim(0, 0.1)
            ax[i].set_yticks(np.arange(0, 0.11, 0.01))
            ax[i].grid(True)
        display.display(fig)
        fig.savefig(folder_result+"/loss_epoch"+str(epoch)+".jpg")
        plt.close(fig)
#%%
model_state_last={'encoder':deepcopy(encoder_net.state_dict()),
                  'decoder':deepcopy(decoder_net.state_dict())}
#%%
encoder_net.load_state_dict(model_state_best['encoder'])
decoder_net.load_state_dict(model_state_best['decoder'])
#%%
mrse_train=test_save(X_train, x_train, filelist_train, arg.folder_data, folder_result_train)
mrse_val=test_save(X_val, x_val, filelist_val, arg.folder_data, folder_result_val)
mrse_test=test_save(X_test, x_test, filelist_test, arg.folder_data, folder_result_test)
print("train: mrse", mrse_train)
print("val: mrse", mrse_val)
print("test: mrse", mrse_test)
#%%
if arg.max_epochs > 0:
    torch.save({"arg":arg,
                "encoder_model_state":encoder_net.state_dict(),
                "decoder_model_state":decoder_net.state_dict(),
                "model_state_last":model_state_last,
                "MeanShape":MeanShape,
                "mrse_train":mrse_list_train,
                "mrse_val":mrse_list_val,
                "mrse_test":mrse_list_test},
               filename_save)
    print("saved:", filename_save)
#%%

