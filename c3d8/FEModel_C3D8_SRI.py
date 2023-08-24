# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 19:05:36 2021

@author: liang
"""
import torch
from torch import matmul
import torch_scatter
from Element_C3D8 import get_integration_point_1i, get_integration_point_8i, cal_d_sf_dh, cal_F_tensor_with_d_sf_dX
from Element_C3D8 import cal_nodal_force_from_cauchy_stress_8i, cal_nodal_force_from_1pk_stress_8i
from Element_C3D8 import cal_nodal_force_from_cauchy_stress_1i, cal_nodal_force_from_1pk_stress_1i
from Element_C3D8 import cal_strain_energy_8i, cal_strain_energy_1i
from FEModel_C3D8 import cal_attribute_on_node
from FEModel_C3D8 import cal_F_tensor_8i, cal_F_tensor_1i
from FEModel_C3D8 import cal_d_sf_dx_and_dx_dr_8i, cal_d_sf_dx_and_dx_dr_1i
from FEModel_C3D8 import cal_d_sf_dX_and_dX_dr_8i, cal_d_sf_dX_and_dX_dr_1i
from FEModel_C3D8 import cal_pressure_force_quad_4i, cal_pressure_force_quad_1i
from FEModel_C3D8 import cal_dense_stiffness_matrix, cal_sparse_stiffness_matrix, cal_diagonal_stiffness_matrix
#%%
def cal_cauchy_stress_force(node_x, element, d_sf_dX_8i, d_sf_dX_1i, material, cal_cauchy_stress,
                            F0_8i=None, F0_1i=None,
                            return_F_S_W=False, return_stiffness=None, return_force_of_element=False):
    #d_sf_dX_8i from cal_d_sf_dX_and_dX_dr_8i
    #d_sf_dX_1i from cal_d_sf_dX_and_dX_dr_1i
    x=node_x[element]
    F_8i=cal_F_tensor_with_d_sf_dX(x, d_sf_dX_8i)
    F_1i=cal_F_tensor_with_d_sf_dX(x, d_sf_dX_1i)
    if F0_8i is not None and F0_1i is not None:
        #F0 is residual/pre deformation
        F_8i=matmul(F_8i, F0_8i)
        F_1i=matmul(F_1i, F0_1i)
    Sd, Sv, Wd, Wv=cal_cauchy_stress(F_8i, F_1i, material, create_graph=True, return_W=True)
    #Sd.shape: (M,8,3,3), Sv.shape: (M,1,3,3)
    #-----------------------------------------------------------------------------
    r_8i=get_integration_point_8i(dtype=node_x.dtype, device=node_x.device)
    d_sf_dx_8i, dx_dr_8i, det_dx_dr_8i=cal_d_sf_dh(r_8i, x)
    force_element_Sd=cal_nodal_force_from_cauchy_stress_8i(Sd, d_sf_dx_8i, det_dx_dr_8i)
    #force_element_8i.shape: (M,8,3)
    #-----------------------------------------------------------------------------
    r_1i=get_integration_point_1i(dtype=node_x.dtype, device=node_x.device)
    d_sf_dx_1i, dx_dr_1i, det_dx_dr_1i=cal_d_sf_dh(r_1i, x)
    force_element_Sv=cal_nodal_force_from_cauchy_stress_1i(Sv, d_sf_dx_1i, det_dx_dr_1i)
    #force_element_1i.shape: (M,8,3)
    #-----------------------------------------------------------------------------
    force_element=force_element_Sd+force_element_Sv
    N=node_x.shape[0]
    #M=element.shape[0]
    force = torch_scatter.scatter(force_element.view(-1,3), element.view(-1), dim=0, dim_size=N, reduce="sum")
    #force.shape: (N, 3)
    #-----------------------------------------------------------------------------
    if return_stiffness =="none" or return_stiffness is None:
        return_stiffness = None
    elif return_stiffness == "dense":
        H=cal_dense_stiffness_matrix(N, element, force_element, x)
    elif return_stiffness == "sparse":
        H=cal_sparse_stiffness_matrix(N, element, force_element, x)
    elif return_stiffness == "diagonal":
        H=cal_diagonal_stiffness_matrix(N, element, force_element, x)
    else:
        raise ValueError("return_stiffness unkown")
    #-----------------------------------------------------------------------------
    #out=[force, F_8i, F_1i, Sd, Sv, Wd, Wv, H, force_element]
    out=[force]
    if return_F_S_W == True:
        out.extend([F_8i, F_1i, Sd, Sv, Wd, Wv])
    if return_stiffness is not None:
        out.append(H)
    if return_force_of_element == True:
        out.append(force_element)
    return out
#%%
def cal_cauchy_stress_force_inverse(d_sf_dx_8i, d_sf_dx_1i, det_dx_dr_8i, det_dx_dr_1i,
                                    n_nodes, element, F_8i, F_1i, material, cal_cauchy_stress,
                                    F0_8i=None, F0_1i=None, return_S_W=False):
    if F0_8i is not None and F0_1i is not None:
        #F0 is residual/pre deformation
        F_8i=matmul(F_8i, F0_8i)
        F_1i=matmul(F_1i, F0_1i)
    Sd, Sv, Wd, Wv=cal_cauchy_stress(F_8i, F_1i, material, create_graph=True, return_W=True)
    #Sd.shape: (M,8,3,3), Sv.shape: (M,1,3,3)
    #-----------------------------------------------------------------------------
    force_element_Sd=cal_nodal_force_from_cauchy_stress_8i(Sd, d_sf_dx_8i, det_dx_dr_8i)
    #force_element_8i.shape: (M,8,3)
    #-----------------------------------------------------------------------------
    force_element_Sv=cal_nodal_force_from_cauchy_stress_1i(Sv, d_sf_dx_1i, det_dx_dr_1i)
    #force_element_1i.shape: (M,8,3)
    #-----------------------------------------------------------------------------
    force_element=force_element_Sd+force_element_Sv
    N=n_nodes
    #M=element.shape[0]
    force = torch_scatter.scatter(force_element.view(-1,3), element.view(-1), dim=0, dim_size=N, reduce="sum")
    #force.shape: (N, 3)
    #-----------------------------------------------------------------------------
    if return_S_W == False:
        return force
    else:
        return force, Sd, Sv, Wd, Wv
#%%
def cal_1pk_stress_force(node_x, element, d_sf_dX_8i, d_sf_dX_1i, det_dX_dr_8i, det_dX_dr_1i,
                         material, cal_1pk_stress, F0_8i=None, F0_1i=None,
                         return_F_S_W=False, return_stiffness=None, return_force_of_element=False):
    x=node_x[element]
    F_8i=cal_F_tensor_with_d_sf_dX(x, d_sf_dX_8i)
    F_1i=cal_F_tensor_with_d_sf_dX(x, d_sf_dX_1i)
    if F0_8i is not None and F0_1i is not None:
        #F0 is residual/pre deformation
        F_8i=matmul(F_8i, F0_8i)
        F_1i=matmul(F_1i, F0_1i)
    #F_8i.shape: (M,8,3,3)
    #F_1i.shape: (M,1,3,3)
    #material.shape: (M,A) or (1,A)
    Sd, Sv, Wd, Wv=cal_1pk_stress(F_8i, F_1i, material, create_graph=True, return_W=True)
    #Sd.shape: (M,8,3,3), Sv.shape: (M,1,3,3)
    force_element_Sd=cal_nodal_force_from_1pk_stress_8i(Sd, d_sf_dX_8i, det_dX_dr_8i)
    force_element_Sv=cal_nodal_force_from_1pk_stress_1i(Sv, d_sf_dX_1i, det_dX_dr_1i)
    force_element=force_element_Sd+force_element_Sv
    #force_element.shape: (M,8,3)
    #-----------------------------------------------------------------------------
    N=node_x.shape[0]
    #M=element.shape[0]
    force = torch_scatter.scatter(force_element.view(-1,3), element.view(-1), dim=0, dim_size=N, reduce="sum")
    #force.shape: (N, 3)
    #-----------------------------------------------------------------------------
    if return_stiffness =="none" or return_stiffness is None:
        return_stiffness = None
    elif return_stiffness == "dense":
        H=cal_dense_stiffness_matrix(N, element, force_element, x)
    elif return_stiffness == "sparse":
        H=cal_sparse_stiffness_matrix(N, element, force_element, x)
    elif return_stiffness == "diagonal":
        H=cal_diagonal_stiffness_matrix(N, element, force_element, x)
    else:
        raise ValueError("return_stiffness unkown")
    #-----------------------------------------------------------------------------
    #out=[force, F_8i, F_1i, Sd, Sv, Wd, Wv, H, force_element]
    out=[force]
    if return_F_S_W == True:
        out.extend([F_8i, F_1i, Sd, Sv, Wd, Wv])
    if return_stiffness is not None:
        out.append(H)
    if return_force_of_element == True:
        out.append(force_element)
    return out
#%%
def cal_strain_energy(Wd, Wv, det_dX_dr_8i, det_dX_dr_1i, reduction='sum'):
    energy_8i=cal_strain_energy_8i(Wd, det_dX_dr_8i, reduction)
    energy_1i=cal_strain_energy_1i(Wv, det_dX_dr_1i, reduction)
    if reduction == 'sum':
        energy=energy_8i+energy_1i
    elif reduction == 'mean':
        energy=(energy_8i+energy_1i)/2
    else:
        raise ValueError("not supported reduction:"+reduction)
    return energy
