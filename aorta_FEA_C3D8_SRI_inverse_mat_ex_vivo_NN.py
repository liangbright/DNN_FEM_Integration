import sys
sys.path.append("./c3d8")
sys.path.append("./mesh")
import numpy as np
from IPython import display
import matplotlib.pyplot as plt
import torch
from AortaFEModel_C3D8_SRI import AortaFEModel
from FEModel_C3D8 import cal_attribute_on_node
from MatNet import Net0, Net3
from PolyhedronMesh import PolyhedronMesh
import time
from lbfgs import LBFGS
from aorta_mesh import get_solid_mesh_cfg
from aorta_mat_distribution import generate_mat_distribution
#%%
import argparse
parser = argparse.ArgumentParser(description='Input Parameters:')
parser.add_argument('--cuda', default=0, type=int)
parser.add_argument('--folder', default='./result/inverse/mat_distribution2_mean_shape/', type=str)
parser.add_argument('--mesh_p0', default='p0_171_solid', type=str)
parser.add_argument('--mesh_px', default='p0_171_solid_mat_distribution2_p18', type=str)
parser.add_argument('--shell_template', default='bav17_AortaModel_P0_best', type=str)
#parser.add_argument('--mesh_input', default='aorta_tube_solid_1layers', type=str)
parser.add_argument('--mesh_input', default='p0_171_solid', type=str)#template
parser.add_argument('--mat_distribution', default='2', type=str)
parser.add_argument('--pressure', default=18, type=float)
parser.add_argument('--max_iter', default=100000, type=int)
parser.add_argument('--net', default="Net3(3,256,4,1,1,1,5)", type=str)
#parser.add_argument('--net', default="Net0(3,256,4,1,1,1,5)", type=str)
#parser.add_argument('--net', default="none", type=str)
arg = parser.parse_args()
print(arg)
#%%
device=torch.device("cuda:"+str(arg.cuda))
#device=torch.device("cpu")
dtype=torch.float64
#%%
filename_shell=arg.folder+arg.shell_template+".pt"
(boundary0, boundary1, Element_surface_pressure, Element_surface_free)=get_solid_mesh_cfg(filename_shell, n_layers=1)
#%%
Mesh_X=PolyhedronMesh()
Mesh_X.load_from_torch(arg.folder+arg.mesh_p0+".pt")
Node_X=Mesh_X.node.to(dtype).to(device)
Element=Mesh_X.element.to(device)
#%%
Mesh_x=PolyhedronMesh()
Mesh_x.load_from_torch(arg.folder+arg.mesh_px+".pt")
Node_x=Mesh_x.node.to(dtype).to(device)
#%%
if len(arg.mesh_input)>0:
    mesh_input=PolyhedronMesh()
    #mesh_input.load_from_vtk(arg.folder+arg.mesh_input+".vtk", dtype=dtype)
    mesh_input.load_from_torch(arg.folder+arg.mesh_input+".pt")
else:
    mesh_input=Mesh_X
    print('Mesh_X is mesh_input')
if arg.mesh_input == arg.mesh_p0:
    print('Mesh_X is mesh_input')
NodeInput=mesh_input.node[mesh_input.element].mean(dim=1)
NodeInput=NodeInput.to(dtype).to(device)
#%%
from Mat_GOH_SRI import cal_1pk_stress, cal_cauchy_stress
def process_raw_mat(Mat_raw):
    m0_min=1;    m0_max=1000
    m1_min=0;    m1_max=5500
    m2_min=0.01; m2_max=50
    M=torch.sigmoid(Mat_raw)
    Mat=torch.zeros((Element.shape[0], 6), dtype=dtype, device=device)
    Mat[:,0]=m0_min+(m0_max-m0_min)*M[:,0]
    Mat[:,1]=m1_min+(m1_max-m1_min)*M[:,1]
    Mat[:,2]=m2_min+(m2_max-m2_min)*M[:,2]
    Mat[:,3]=(1/3)*M[:,3]
    Mat[:,4]=(np.pi/2)*M[:,4]
    Mat[:,5]=1e5 # a known constant
    return Mat
#%%
with torch.no_grad():
    Mat_init=torch.zeros((Element.shape[0], 5), dtype=dtype, device=device, requires_grad=True)
    Mat_init=process_raw_mat(Mat_init)
#%%
aorta_model=AortaFEModel(Node_x, Element, Node_X, boundary0, boundary1, Element_surface_pressure,
                         Mat_init, cal_1pk_stress, cal_cauchy_stress, dtype, device, mode='inverse_mat')
pressure=arg.pressure
#%% NN or RawMat to solve for the mat field
if arg.net != "none":
    mat_net=eval(arg.net).to(dtype).to(device)
else:
    RawMat=torch.zeros((Element.shape[0], 5), dtype=dtype, device=device, requires_grad=True)
#%%
def run_mat_net():
    Mat_raw=mat_net(NodeInput)
    Mat=process_raw_mat(Mat_raw)
    return Mat
#%%
def cal_mat():
    if arg.net !="none":
        Mat=run_mat_net()
    else:
        Mat=process_raw_mat(RawMat)
    return Mat
#%%
def loss_function(A, B, reduction):
    Res=A-B
    if reduction == "MSE":
        loss=(Res**2).mean()
    elif reduction == "RMSE":
        loss=(Res**2).mean().sqrt()
    elif reduction == "MAE":
        loss=Res.abs().mean()
    elif reduction == "SSE":
        loss=(Res**2).sum()
    elif reduction == "SAE":
        loss=Res.abs().sum()
    return loss
#%%
print("initilization with Mat_init")
print(Mat_init[0])
if arg.net !="none":
    optimizer = LBFGS(mat_net.parameters(), lr=1, line_search_fn="strong_wolfe")
    for iter1 in range(0, 100):
        def closure():
            Mat=run_mat_net()
            loss=((Mat-Mat_init)**2).sum()
            if loss.requires_grad==True:
                optimizer.zero_grad()
                loss.backward()
            return loss
        optimizer.step(closure)
        if iter1%10==0:
            Mat=run_mat_net()
            Mat_mean=Mat.mean(dim=0).detach().cpu().numpy().tolist()
            print(iter1, Mat_mean)
#%%
loss_list=[]
error_list=[]
t_list=[]
t0=time.time()
#%%
if arg.net !="none":
    optimizer = LBFGS(mat_net.parameters(), lr=1, line_search_fn="strong_wolfe")
else:
    optimizer = LBFGS([RawMat], lr=1, line_search_fn="strong_wolfe")
#%%
Mat_true=generate_mat_distribution(int(arg.mat_distribution), arg.folder)
Mat_true=Mat_true.clone()
Mat_true[:,4]=np.pi*(Mat_true[:,4]/180)
Mat_true=Mat_true.to(device)
#%%
def save_Mat(save_to_p0_or_px):
    Mat_element=cal_mat()
    Mat_element_true=Mat_true.expand(Element.shape[0],-1)
    Mat_node_true=cal_attribute_on_node(Node_X.shape[0], Element, Mat_element_true)
    Mat_element_true=Mat_element_true.detach().cpu()
    Mat_node=cal_attribute_on_node(Node_X.shape[0], Element, Mat_element)
    Mat_element=Mat_element.detach().cpu()
    Mat_node=Mat_node.detach().cpu()
    Mat_node_true=Mat_node_true.detach().cpu()
    Mesh_mat=PolyhedronMesh()
    if save_to_p0_or_px == 'p0':
        Mesh_mat.node=Node_X.detach().cpu()
    else:
        Mesh_mat.node=Node_x.detach().cpu()
    Mesh_mat.element=Element.detach().cpu()
    Mesh_mat.node_data['Mat_true']=Mat_node_true
    Mesh_mat.node_data['Mat_pred']=Mat_node
    Mesh_mat.node_data['Error']=(Mat_node-Mat_node_true).abs()/Mat_node_true.mean(dim=0, keepdim=True)
    Mesh_mat.element_data['Mat_true']=Mat_element_true
    Mesh_mat.element_data['Mat_pred']=Mat_element
    Mesh_mat.element_data['Error']=(Mat_element-Mat_element_true).abs()/Mat_element_true.mean(dim=0, keepdim=True)
    try:
        Mesh_mat.mesh_data['model_state']=mat_net.state_dict()
    except:
        pass
    Mesh_mat.mesh_data['loss']=torch.tensor(loss_list)
    Mesh_mat.mesh_data['memory']=torch.cuda.memory_stats(device=device)
    Mesh_mat.mesh_data['time']=torch.tensor(t_list)
    Mesh_mat.mesh_data['error']=torch.tensor(error_list)
    if save_to_p0_or_px == 'p0':
        filename_save=arg.folder+arg.mesh_px+"_p0_mat_net_"+arg.net # do not use arg.mesh_p0
    else:
        filename_save=arg.folder+arg.mesh_px+"_mat_net_"+arg.net
    Mesh_mat.save_by_vtk(filename_save+".vtk")
    Mesh_mat.save_by_torch(filename_save+".pt")
    print("save", filename_save)
#%%
print("run nonuniform estimation")
if 1:
    for iter1 in range(0, arg.max_iter):
        def closure(loss_fn="SSE"):
            Mat=cal_mat()
            aorta_model.set_material(Mat)
            out =aorta_model.cal_energy_and_force(pressure)
            force_int=out['force_int']
            force_ext=out['force_ext']
            loss=loss_function(force_int, force_ext, loss_fn)
            if loss.requires_grad==True:
                optimizer.zero_grad()
                loss.backward()
            return loss
        opt_cond=optimizer.step(closure)
        #
        loss=closure(loss_fn="RMSE")
        loss=float(loss)
        loss_list.append(loss)
        t1=time.time()
        t_list.append(t1-t0)
        #
        if np.isnan(loss) == True or np.isinf(loss) == True:
            break
        #
        Mat=cal_mat()
        Mat_mean=Mat.mean(dim=0).detach().cpu().numpy().tolist()
        Error=(Mat-Mat_true).abs()/Mat_true.mean(dim=0, keepdim=True)
        Error_mean=Error.mean(dim=0).detach().cpu().numpy().tolist()
        Error_max=Error.max(dim=0)[0].detach().cpu().numpy().tolist()
        Error_min=Error.min(dim=0)[0].detach().cpu().numpy().tolist()
        error_list.append([Error_mean, Error_max, Error_min])
        #
        if (iter1)%100 == 0 or opt_cond == True:
            print(iter1, loss, t1-t0)
            print("mat_mean", Mat_mean)
            print("error_mean", Error_mean)
            print("error_max", Error_max)
            print("error_min", Error_min)
            display.clear_output(wait=False)
            fig, ax = plt.subplots()
            ax.plot(np.array(loss_list), 'r')
            ax.set_ylim(0, 0.1)
            ax.grid(True)
            display.display(fig)
            plt.close(fig)
        #
        if opt_cond == True:
            print("opt_cond is True, break")
            break
    #---------
    save_Mat(save_to_p0_or_px='px')
    save_Mat(save_to_p0_or_px='p0')

