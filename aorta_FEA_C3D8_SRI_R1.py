import sys
sys.path.append("./c3d8")
sys.path.append("./mesh")
import numpy as np
from IPython import display
import matplotlib.pyplot as plt
import torch
from torch.linalg import det
from AortaFEModel_C3D8_SRI import AortaFEModel, cal_potential_energy, cal_element_orientation
from FEModel_C3D8_SRI_fiber import cal_attribute_on_node
from FEModel_C3D8 import cal_F_tensor_8i, cal_F_tensor_1i
from von_mises_stress import cal_von_mises_stress
from PolyhedronMesh import PolyhedronMesh
from aorta_mesh import get_solid_mesh_cfg
import time
#%%
matA="200, 0, 1, 0.3333, 0, 1e5"
matB="50, 1000, 10, 0.3333, 0, 1e5"
matC="50, 1000, 10, 0.1, 60, 1e5"
mat_sd="1e5, 0, 1, 0.3333, 0, 1e5"
all_mat=torch.load('./data/125mat.pt')['mat_str']
mat95=all_mat[95]
mat10=all_mat[10]
mat24=all_mat[24]
mat64=all_mat[64]
matMean=torch.load('./data/125mat.pt')['mean_mat_str']
#%%
folder_data='./data/'
folder_result='./result/forward/'
#%%
import argparse
parser = argparse.ArgumentParser(description='Input Parameters:')
parser.add_argument('--cuda', default=0, type=int)
parser.add_argument('--dtype', default="float64", type=str)
parser.add_argument('--shell_template', default=folder_data+'bav17_AortaModel_P0_best.pt', type=str)
parser.add_argument('--mesh_p0', default=folder_data+'p0_47_solid', type=str)
parser.add_argument('--mesh_px_pred', default=folder_result+'p0_47_solid_matMean_p18_pred_refine', type=str)
parser.add_argument('--mesh_px_init', default=folder_result+'p0_47_solid_matMean_p18_pred', type=str)
#parser.add_argument('--mesh_px_init', default='', type=str)
parser.add_argument('--mat', default=matMean, type=str)
parser.add_argument('--pressure', default=18, type=float)
parser.add_argument('--max_iter1', default=1, type=int)
parser.add_argument('--max_iter2', default=1000, type=int)
parser.add_argument('--use_stiffness', default=True, type=bool)
parser.add_argument('--reform_stiffness_interval', default=20, type=int)
parser.add_argument('--warm_up_T0', default=10, type=int)
parser.add_argument('--ratio_Interval', default=10, type=int)
parser.add_argument('--TPE_ratio', default=0.01, type=float)
parser.add_argument('--loss1', default=0.01, type=float)
parser.add_argument('--Rmax', default=0.005, type=float)
parser.add_argument('--save_by_vtk', default='True', type=str)#True: save *.vtk file in addition to *.pt file
parser.add_argument('--plot', default='False', type=str)
arg = parser.parse_args()
arg.save_by_vtk = True if arg.save_by_vtk == 'True' else False
arg.plot = True if arg.plot == 'True' else False
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
n_layers=1
if '4layer' in arg.mesh_p0:
    n_layers=4
(boundary0, boundary1, Element_surface_pressure, Element_surface_free)=get_solid_mesh_cfg(arg.shell_template, n_layers)
#%%
Mesh_X=PolyhedronMesh()
Mesh_X.load_from_torch(arg.mesh_p0+".pt")
Node_X=Mesh_X.node.to(dtype).to(device)
Element=Mesh_X.element.to(device)
#%%
from Mat_GOH_SRI import cal_1pk_stress, cal_cauchy_stress
#from InvivoMat_GOH_SRI import cal_1pk_stress, cal_cauchy_stress
Mat=[float(m) for m in arg.mat.split(",")]
Mat[4]=np.pi*(Mat[4]/180)
Mat=torch.tensor([Mat], dtype=dtype, device=device)
aorta_model=AortaFEModel(None, Element, Node_X, boundary0, boundary1, Element_surface_pressure,
                         Mat, cal_1pk_stress, cal_cauchy_stress, dtype, device, mode='inflation')
aorta_model.element_orientation_on_deformed_mesh=False
pressure=arg.pressure
#%%
def loss1a_function(force_int, force_ext):
    loss=((force_int-force_ext)**2).sum(dim=1).sqrt().mean()
    return loss
#%%
def loss1b_function(force_int, force_ext):
    loss=((force_int-force_ext)**2).sum()
    return loss
#%%
if len(arg.mesh_px_init) > 0:
    Mesh_x_init=PolyhedronMesh()
    Mesh_x_init.load_from_torch(arg.mesh_px_init+".pt")
    Node_x_init=Mesh_x_init.node.to(dtype).to(device)
    u_field_init=Node_x_init-Node_X
    u_field_init=u_field_init*aorta_model.state['mask']
    print("initilize u_field with u_field_init")
    aorta_model.state['u_field'].data=u_field_init[aorta_model.state['free_node']].to(dtype).to(device)
    out=aorta_model.cal_energy_and_force(pressure)
    TPE1=out['TPE1']; TPE2=out['TPE2']; SE=out['SE']
    force_int=out['force_int']; force_ext=out['force_ext']
    force_int_of_element=out['force_int_of_element']
    F=out['F']; u_field=out['u_field']
    loss1=loss1a_function(force_int, force_ext)
    loss1=loss1.item()
    force_avg=(force_int_of_element**2).sum(dim=2).sqrt().mean()
    force_res=((force_int-force_ext)**2).sum(dim=1).sqrt()
    R=force_res/(force_avg+1e-10)
    Rmean=R.mean().item()
    Rmax=R.max().item()
    J=det(F)
    print("init: pressure", pressure)
    print("init: TPE", float(TPE1), "loss1", loss1, "Rmax", Rmax, "max|J-1|", float((J-1).abs().max()))
    sum_force_res=force_res.sum().item()
    print("init: sum(force_res)", force_res.sum().item(), "1/sum(force_res)", 1/sum_force_res)
    #1/sum_force_res should be similar to t_default for the first step
    if loss1 < arg.loss1 and Rmax < arg.Rmax:
        opt_cond=True
        arg.max_iter1=0
        print("mesh_px_init is optimal, set arg.max_iter1=0, set opt_cond=True")
#%%
    del Mesh_x_init, Node_x_init, u_field_init
    del out, TPE1, TPE2, SE, force_int, force_ext, J, F, u_field, R, force_res, force_avg, force_int_of_element
#%%
loss1_list=[]
loss2_list=[]
Rmax_list=[]
TPE_list=[]
time_list=[]
flag_list=[]
t0=time.time()
#%%
from FE_lbfgs import LBFGS
optimizer = LBFGS([aorta_model.state['u_field']], lr=1, line_search_fn="backtracking",
                  tolerance_grad =1e-10, tolerance_change=1e-20, history_size =100, max_iter=1)
optimizer.set_backtracking(c=0.5, verbose=False)
t_listA=[1, 0.5, 0.1, 0.05, 0.01]
t_listB=[0.5, 0.3, 0.1, 0.05, 0.01]
#%%
def reform_stiffness(pressure_t, iter2):
    ta=time.time()
    Output=aorta_model.cal_energy_and_force(pressure_t, return_stiffness="sparse")
    H=Output['H']
    H=H.astype("float64")
    optimizer.reset_state()
    optimizer.set_H0(H)
    tb=time.time()
    print("reform stiffness done:", tb-ta, "iter2", iter2)
    return tb-ta
#%%
max_iter1=arg.max_iter1
max_iter2=arg.max_iter2
Ugood=aorta_model.state['u_field'].clone().detach()
closure_opt={"output":"TPE", "loss1_fn":loss1a_function}
#%%
for iter1 in range(0, max_iter1):
    optimizer.reset_state()
    pressure_iter=((iter1+1)/max_iter1)*pressure
    print("pressure_iter", pressure_iter)

    error_flag=False
    flag_use_stiffness=False

    optimizer.reset_state()
    optimizer.set_backtracking(t_list=t_listA, t_default=0.5, t_default_init="auto")
    #t_default_init for the first step must be very small to prevent nan

    for iter2 in range(0, max_iter2):
        #------------------------------------------------
        def closure(output=closure_opt["output"], loss1_fn=closure_opt["loss1_fn"]):
            out=aorta_model.cal_energy_and_force(pressure_iter)
            TPE1=out['TPE1']; TPE2=out['TPE2']; SE=out['SE']
            force_int=out['force_int']; force_ext=out['force_ext']
            F=out['F']; u_field=out['u_field']
            loss1=loss1_fn(force_int, force_ext)
            force_res=force_int-force_ext
            loss2=cal_potential_energy(force_res, u_field)
            if output == "TPE":
                loss=loss2
            elif output == "loss1":
                loss=loss1
            elif output == "TPE+loss1":
                loss=loss2+loss1
            else:
                loss=loss2
            if loss.requires_grad==True:
                optimizer.zero_grad()
                loss.backward()
            if output == "TPE":
                return TPE1
            elif output == "loss1":
                return loss1
            elif output == "TPE+loss1":
                return TPE+loss1
            else:
                loss1=float(loss1)
                loss2=float(loss2)
                TPE1=float(TPE1)
                F=F.detach()
                force_int=force_int.detach()
                force_ext=force_ext.detach()
                force_int_of_element=out['force_int_of_element'].detach()
                return loss1, loss2, TPE1, F, force_int, force_ext, force_int_of_element
        #------------------------------------------------
        opt_cond=optimizer.step(closure)
        flag_linesearch=optimizer.get_linesearch_flag()
        t_linesearch=optimizer.get_linesearch_t()
        #------------------------------------------------
        loss1, loss2, TPE, F, force_int, force_ext, force_int_of_element=torch.no_grad()(closure)(output='all')
        J=det(F)
        #------------------------------------------------
        #check error
        error_flag_counter=0
        #TPE is nan or inf?
        if (np.isnan(TPE) == True or np.isinf(TPE) == True
            or np.isnan(loss1) == True or np.isinf(loss1) == True
            or np.isnan(loss2) == True or np.isinf(loss2) == True):
            opt_cond=False
            error_flag=True
            error_flag_counter+=1
            print(iter1, iter2, "error: TPE", TPE, "loss1", loss1, "loss2", loss2, "max|J-1|", float((J-1).abs().max()))
            print("flag", flag_linesearch, "t_linesearch", t_linesearch)
            aorta_model.state['u_field'].data=Ugood.clone().detach()
            optimizer.reset_state()
            optimizer.set_backtracking(t_list=t_listA, t_default=0.5, t_default_init="auto")
            flag_use_stiffness=False
            if iter2==0:
                print("error at iter2 = 0, break")
                break
            continue
        #It is possible that TPE > 0, not as bad as TPE=nan or +/- inf
        if TPE > 0:
             opt_cond=False
             error_flag=True
             error_flag_counter+=1
             #print(iter1, iter2, "error: TPE is", TPE, "loss1 is", loss1, ", max|J-1| =", float((J-1).abs().max()))
             #print("flag", flag_linesearch, "t_linesearch", t_linesearch)
             if flag_use_stiffness == True:
                 aorta_model.state['u_field'].data=Ugood.clone().detach()
                 optimizer.reset_state()
                 optimizer.set_backtracking(t_list=t_listA, t_default=0.5, t_default_init=0.5)
                 flag_use_stiffness=False
             continue
        #some elements may deform too much
        J_error_counter=((J-1).abs()>0.5).sum().item()
        if J_error_counter > 0:
            opt_cond=False
            error_flag=True
            error_flag_counter+=1
            if flag_use_stiffness == True:
                aorta_model.state['u_field'].data=Ugood.clone().detach()
                optimizer.reset_state()
                optimizer.set_backtracking(t_list=t_listA, t_default=0.5, t_default_init=0.5)
                flag_use_stiffness=False
            if iter2 > arg.warm_up_T0:
                print(iter1, iter2, "error: sum(abs(J-1)>0.5) =", J_error_counter)
                print("flag", flag_linesearch, "t_linesearch", t_linesearch)
            continue
        #set error_flag
        if error_flag_counter == 0:
            error_flag=False
        #------------------------------------------------
        if iter2 >= 1 and error_flag==False and flag_use_stiffness == False:
            optimizer.set_backtracking(t_list=t_listA, t_default=0.5, t_default_init=0.5)
        #------------------------------------------------

        flag_list.append(flag_linesearch)

        t1=time.time()
        time_list.append(t1-t0)

        TPE_list.append(TPE)
        loss1_list.append(loss1)
        loss2_list.append(loss2)

        force_avg=(force_int_of_element**2).sum(dim=2).sqrt().mean()
        force_res=((force_int-force_ext)**2).sum(dim=1).sqrt()
        R=force_res/(force_avg+1e-10)
        Rmean=R.mean().item()
        Rmax=R.max().item()
        Rmax_list.append(Rmax)
        #-------------------------------------------------------
        if error_flag == False and len(TPE_list) > 1:
            if TPE < min(TPE_list[:-1]) or loss1 < min(loss1_list[:-1]) or Rmax < min(Rmax_list[:-1]):
                Ugood=aorta_model.state['u_field'].clone().detach()
        #-------------------------------------------------------
        T=arg.ratio_Interval
        T0=max(arg.warm_up_T0, T)
        if len(TPE_list) < T0:
            TPE_ratio=1
            loss1_ratio=1
            if len(TPE_list) > 2:
                id_T=max(-T, -len(TPE_list))
                TPE_ratio=abs(TPE_list[-1]-TPE_list[id_T])/(1e-10+abs(TPE_list[-1]))
                loss1_ratio=abs(loss1_list[-1]-loss1_list[id_T])/(1e-10+abs(loss1_list[0]))
        else:
            TPE_ratio=abs(TPE_list[-1]-TPE_list[-T])/(1e-10+abs(TPE_list[-1]))
            loss1_ratio=abs(loss1_list[-1]-loss1_list[-T])/(1e-10+abs(loss1_list[0]))
            if error_flag == False and flag_use_stiffness == False and arg.use_stiffness==True:
                if TPE_ratio < arg.TPE_ratio:  #loss1_ratio is not good for input close to optimal (loss1 very small)
                    flag_use_stiffness=True
                    print("iter1", iter1, "iter2", iter2, "pressure_iter", pressure_iter, "time", t1-t0)
                    print("flag", flag_linesearch, "t_linesearch", t_linesearch, "TPE_ratio", TPE_ratio, "loss1_ratio", loss1_ratio)
                    print('Rmax', Rmax, "loss1", loss1, "loss2", loss2, "TPE", TPE, "max|J-1|", float((J-1).abs().max()))
                    print('set flag_use_stiffness to True')
        #-------------------------------------------------------
        if arg.use_stiffness == True and flag_use_stiffness == True and error_flag == False:
            if iter2%arg.reform_stiffness_interval == 0:
                reform_stiffness(pressure_iter, iter2)
                #loss nan if the solution is not close to the true optimal one
                optimizer.set_backtracking(t_list=t_listB, t_default=0.5, t_default_init=0.5)
        #------------------------------------------
        opt_cond=False
        #check convergance
        if loss1 < arg.loss1 and Rmax < arg.Rmax:
            opt_cond=True
        #------------------------------------------
        flag_print=False
        if iter2==0:
            flag_print=True
        if flag_use_stiffness==False and iter2%T == 0:
           flag_print=True
        if arg.use_stiffness==True and flag_use_stiffness==True and (iter2+1)%arg.reform_stiffness_interval==0:
            flag_print=True
        if flag_print==True or opt_cond == True:
            print("iter1", iter1, "iter2", iter2, "pressure_iter", pressure_iter, "time", t1-t0)
            print("flag", flag_linesearch, "t_linesearch", t_linesearch, "TPE_ratio", TPE_ratio, "loss1_ratio", loss1_ratio)
            print('Rmax', Rmax, "loss1", loss1, "loss2", loss2, "TPE", TPE, "max|J-1|", float((J-1).abs().max()))
            if arg.plot == True:
                display.clear_output(wait=False)
                fig, ax = plt.subplots()
                ax.plot(np.array(loss1_list)/max(loss1_list), 'r')
                ax.plot(loss1_list, 'm')
                if len(TPE_list) > 10:
                    negTPE=-np.array(TPE_list)
                    Vmax=np.max(negTPE[(np.isinf(negTPE)==False)&(negTPE>0)])
                    ax.plot(negTPE/Vmax, 'b')
                ax.plot(-np.array(flag_list), 'g.')
                ax.set_ylim(0, 1)
                ax.grid(True)
                display.display(fig)
                plt.close(fig)
        #-------------------------------------
        if opt_cond == True:
            print("opt_cond is True, break iter2 =", iter2)
            Ugood=aorta_model.state['u_field'].clone().detach()
            error_flag=False
            break
#%%
aorta_model.state['u_field']=Ugood
u_field=aorta_model.get_u_field()
Node_x=Node_X+u_field
S=aorta_model.cal_stress('cauchy', create_graph=False, return_W=False)
S_element=S.mean(dim=1)
VM_element=cal_von_mises_stress(S_element)
S_node=cal_attribute_on_node(Node_x.shape[0], Element, S_element)
VM_node=cal_von_mises_stress(S_node)
Mesh_x=PolyhedronMesh()
Mesh_x.element=Element.detach().cpu()
Mesh_x.node=Node_x.detach().cpu()
Mesh_x.element_data['S']=S_element.view(-1,9)
Mesh_x.element_data['VM']=VM_element.view(-1,1)
Mesh_x.node_data['S']=S_node.view(-1,9)
Mesh_x.node_data['VM']=VM_node.view(-1,1)
Mesh_x.mesh_data['arg']=arg
Mesh_x.mesh_data['TPE']=TPE_list
Mesh_x.mesh_data['loss1']=loss1_list
Mesh_x.mesh_data['Rmax']=Rmax
Mesh_x.mesh_data['time']=time_list
Mesh_x.mesh_data['opt_cond']=opt_cond
mesh_px_pred=arg.mesh_px_pred
if opt_cond == False:
    mesh_px_pred=arg.mesh_px_pred+'_error'
Mesh_x.save_by_torch(mesh_px_pred+".pt")
if arg.save_by_vtk == True:
    Mesh_x.save_by_vtk(mesh_px_pred+".vtk")
print("saved", mesh_px_pred)
