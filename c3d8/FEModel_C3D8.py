# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 19:05:36 2021

@author: liang
"""
import torch
from torch import matmul
import numpy as np
import torch_scatter
import torch_sparse
from Element_C3D8 import interpolate, cal_d_sf_dh
from Element_C3D8 import cal_F_tensor, cal_F_tensor_with_d_sf_dX, cal_F_tensor_X_cube
from Element_C3D8 import get_integration_point_1i, get_integration_point_8i
from Element_C3D8 import cal_nodal_force_from_cauchy_stress_1i, cal_nodal_force_from_cauchy_stress_8i
from Element_C3D8 import cal_nodal_force_from_1pk_stress_1i, cal_nodal_force_from_1pk_stress_8i
from Element_C3D8 import cal_nodal_force_from_cauchy_stress, cal_nodal_force_from_1pk_stress
from Element_C3D8 import cal_strain_energy
from pressure_load_on_quad import cal_nodal_force_from_pressure_1i, cal_nodal_force_from_pressure_4i
#%%
def sample_point(node, element, r=None):
    x=node[element]
    if r is None:
        r=[2*np.random.rand()-1, 2*np.random.rand()-1, 2*np.random.rand()-1]
    x_r=interpolate(r, x)
    return x_r
#%% It works for quad_1i, quad_4i, C3D8, and C3D8R
def cal_dense_stiffness_matrix(n_nodes, element, force_element, x):
    #force_element.shape is (M,4,3) or (M,8,3)
    #x.shape (M,8,3) or (M,4,3)
    H=torch.zeros((3*n_nodes,3*n_nodes), dtype=x[0].dtype, device=x[0].device)
    for n in range(0, x.shape[1]):
        a=3*element[:,n]
        gn0=torch.autograd.grad(force_element[:,n,0].sum(), x, retain_graph=True)[0]
        gn1=torch.autograd.grad(force_element[:,n,1].sum(), x, retain_graph=True)[0]
        gn2=torch.autograd.grad(force_element[:,n,2].sum(), x, retain_graph=True)[0]
        for m in range(0, x.shape[1]):
            g0=gn0[:,m].detach()
            g1=gn1[:,m].detach()
            g2=gn2[:,m].detach()
            b=3*element[:,m]
            H[a, b]+=g0[:,0]
            H[a, b+1]+=g0[:,1]
            H[a, b+2]+=g0[:,2]
            H[a+1, b]+=g1[:,0]
            H[a+1, b+1]+=g1[:,1]
            H[a+1, b+2]+=g1[:,2]
            H[a+2, b]+=g2[:,0]
            H[a+2, b+1]+=g2[:,1]
            H[a+2, b+2]+=g2[:,2]
    H=H.detach()
    return H
#%% return a sparse matrix H
import time
def cal_sparse_stiffness_matrix(n_nodes, element, force_element, x):
    #force_element.shape is (M,8,3) or (M,4,3)
    #x.shape (M,8,3) or (M,4,3)
    RowIndex=[]
    ColIndex=[]
    Value=[]
    #t0=time.time()
    #tab=0
    for n in range(0, x.shape[1]):
        a=3*element[:,n]
        gn0=torch.autograd.grad(force_element[:,n,0].sum(), x, retain_graph=True)[0]
        gn1=torch.autograd.grad(force_element[:,n,1].sum(), x, retain_graph=True)[0]
        gn2=torch.autograd.grad(force_element[:,n,2].sum(), x, retain_graph=True)[0]
        #ta=time.time()
        for m in range(0, x.shape[1]):
            g0=gn0[:,m].detach()
            g1=gn1[:,m].detach()
            g2=gn2[:,m].detach()
            b=3*element[:,m]
            RowIndex.append(a);   ColIndex.append(b);   Value.append(g0[:,0])
            RowIndex.append(a);   ColIndex.append(b+1); Value.append(g0[:,1])
            RowIndex.append(a);   ColIndex.append(b+2); Value.append(g0[:,2])
            RowIndex.append(a+1); ColIndex.append(b);   Value.append(g1[:,0])
            RowIndex.append(a+1); ColIndex.append(b+1); Value.append(g1[:,1])
            RowIndex.append(a+1); ColIndex.append(b+2); Value.append(g1[:,2])
            RowIndex.append(a+2); ColIndex.append(b);   Value.append(g2[:,0])
            RowIndex.append(a+2); ColIndex.append(b+1); Value.append(g2[:,1])
            RowIndex.append(a+2); ColIndex.append(b+2); Value.append(g2[:,2])
        #tb=time.time()
        #tab=tab+tb-ta
    #t1=time.time()
    with torch.no_grad():
        RowIndex=torch.cat(RowIndex, dim=0).view(1,-1)
        ColIndex=torch.cat(ColIndex, dim=0).view(1,-1)
        Index=torch.cat([RowIndex, ColIndex], dim=0)
        Value=torch.cat(Value, dim=0)
        H=torch.sparse_coo_tensor(Index, Value, (3*n_nodes,3*n_nodes))
        H=H.coalesce()
    #t2=time.time()
    #print("cal_sparse_stiffness_matrix", t1-t0, t2-t1, tab)
    return H
#%%
def cal_diagonal_stiffness_matrix(n_nodes, element, force_element, x):
    #force_element.shape is (M,4,3) or (M,8,3)
    #x is a list of x1, x2, ..., nodes per element
    #len(x) is 4 or 8
    t0=time.time()
    H=torch.zeros((3*n_nodes,), dtype=x[0].dtype, device=x[0].device)
    for n in range(0, x.shape[1]):
        g0=torch.autograd.grad(force_element[:,n,0].sum(), x, retain_graph=True)[0].detach()
        g1=torch.autograd.grad(force_element[:,n,1].sum(), x, retain_graph=True)[0].detach()
        g2=torch.autograd.grad(force_element[:,n,2].sum(), x, retain_graph=True)[0].detach()
        a=3*element[:,n]
        H[a]=g0[:,n,0]
        H[a+1]=g1[:,n,1]
        H[a+2]=g2[:,n,2]
    H=H.detach()
    t1=time.time()
    print("cal_diagonal_stiffness_matrix", t1-t0)
    return H
#%%
def cal_pressure_force_quad_1i(pressure, node_x, element, return_force="dense", return_stiffness=None):
    #node_x: all of the nodes of the mesh
    #element.shape: (M, 4), M is the number of elements
    #element could be a subset (surface)
    #--------------------------
    if isinstance(pressure, int) == True or isinstance(pressure, float) == True:
        if pressure == 0:
            if return_force == "dense":
                force=torch.zeros_like(node_x)
            else:
                raise ValueError("return_force unkown")
            if return_stiffness is None:
                return force
            else:
                return force, 0
    #--------------------------
    x=node_x[element].requires_grad_(True)
    #--------------------------
    force_element=cal_nodal_force_from_pressure_1i(pressure, x)
    #force_element.shape is (M,4,3)
    #--------------------------
    N=node_x.shape[0]
    if return_force == "dense":
        force = torch_scatter.scatter(force_element.view(-1,3), element.view(-1), dim=0, dim_size=N, reduce="sum")
        #force.shape is (N,3)
    elif return_force == "sparse":
        row_index=element.view(-1)
        col_index=torch.zeros_like(row_index)
        index, value = torch_sparse.coalesce([row_index, col_index], force_element.view(-1,3), len(row_index), 1, "add")
        row_index, col_index=index
        force=(row_index, value)
        #row_index contains node index in element
        #value.shape is (len(row_index),3)
    else:
        raise ValueError("return_force unkown")
    #--------------------------
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
    #--------------------------
    if return_stiffness is None:
        return force
    else:
        return force, H
#%%
def cal_pressure_force_quad_4i(pressure, node_x, element, return_force="dense", return_stiffness=None):
    #element.shape: (M, 4), M is the number of surface elements
    #element could be a subset
    #--------------------------
    if isinstance(pressure, int) == True or isinstance(pressure, float) == True:
        if pressure == 0:
            if return_force == "dense":
                force=torch.zeros_like(node_x)
            else:
                raise ValueError("return_force unkown")
            if return_stiffness is None:
                return force
            else:
                return force, 0
    #--------------------------
    x=node_x[element].requires_grad_(True)
    #--------------------------
    force_element=cal_nodal_force_from_pressure_4i(pressure, x)
    #force_element.shape is (M,4,3)
    #--------------------------
    N=node_x.shape[0]
    if return_force == "dense":
        force = torch_scatter.scatter(force_element.view(-1,3), element.view(-1), dim=0, dim_size=N, reduce="sum")
        #force.shape is (N,3)
    elif return_force == "sparse":
        row_index=element.view(-1)
        col_index=torch.zeros_like(row_index)
        index, value = torch_sparse.coalesce([row_index, col_index], force_element.view(-1,3), len(row_index), 1, "add")
        row_index, col_index=index
        force=(row_index, value)
        #row_index contains node index in element
        #value.shape is (len(row_index),3)
    else:
        raise ValueError("return_force unkown")
    #--------------------------
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
    #--------------------------
    if return_stiffness is None:
        return force
    else:
        return force, H
#%%
class PotentialEnergy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, force, u):
        ctx.save_for_backward(force, u)
        energy=(force*u).sum()
        return energy
    @staticmethod
    def backward(ctx, grad_output):
        force, u = ctx.saved_tensors
        grad_force = None #grad_output*u
        grad_u = grad_output*force
        return grad_force, grad_u
#---------------------------------
cal_potential_energy=PotentialEnergy.apply
#%%
def cal_F_tensor_1i(node_x, element, node_X, F0=None, d_sf_dX=None):
    #element.shape: (M, 8), M is the number of elements
    #node_X.shape: (N,3), N is the number of nodes, X means undeformed/zero-load
    #node_x.shape: (N,3), N is the number of nodes, x means deformed
    x=node_x[element]
    X=node_X[element]
    if d_sf_dX is None:
        r=get_integration_point_1i(dtype=node_x.dtype, device=node_x.device)
        F=cal_F_tensor(r, x, X)
    else:
        F=cal_F_tensor_with_d_sf_dX(x, d_sf_dX)
    if F0 is not None:
        #F0 is residual deformation
        F=matmul(F, F0)
    #F.shape (M,1,3,3)
    return F
#%%
def cal_F_tensor_8i(node_x, element, node_X, F0=None, d_sf_dX=None):
    #element.shape: (M, 8), M is the number of elements
    #node_X.shape: (N,3), N is the number of nodes, X means undeformed/zero-load
    #node_x.shape: (N,3), N is the number of nodes, x means deformed
    x=node_x[element]
    X=node_X[element]
    if d_sf_dX is None:
        r=get_integration_point_8i(dtype=node_x.dtype, device=node_x.device)
        F=cal_F_tensor(r, x, X)
    else:
        F=cal_F_tensor_with_d_sf_dX(x, d_sf_dX)
    if F0 is not None:
        #F0 is residual deformation
        F=matmul(F, F0)
    #F.shape (M,8,3,3)
    return F
#%%
def cal_F_tensor_1i_X_cube(node_x, element, LX, F0=None):
    #element.shape: (M, 8), M is the number of elements
    #node_x.shape: (N,3), N is the number of nodes, x means deformed
    x=node_x[element]
    r=get_integration_point_1i(dtype=node_x.dtype, device=node_x.device)
    F=cal_F_tensor_X_cube(r, x, LX)
    if F0 is not None:
        #F0 is residual deformation
        F=matmul(F, F0)
    #F.shape (M,1,3,3)
    return F
#%%
def cal_F_tensor_8i_X_cube(node_x, element, LX, F0=None):
    #element.shape: (M, 8), M is the number of elements
    #node_x.shape: (N,3), N is the number of nodes, x means deformed
    x=node_x[element]
    r=get_integration_point_8i(dtype=node_x.dtype, device=node_x.device)
    F=cal_F_tensor_X_cube(r, x, LX)
    if F0 is not None:
        #F0 is residual deformation
        F=matmul(F, F0)
    #F.shape (M,8,3,3)
    return F
#%%
def cal_d_sf_dx_and_dx_dr_1i(node_x, element):
    x=node_x[element]
    r=get_integration_point_1i(dtype=node_x.dtype, device=node_x.device)
    d_sf_dx, dx_dr, det_dx_dr = cal_d_sf_dh(r, x)
    return d_sf_dx, dx_dr, det_dx_dr
#%%
def cal_d_sf_dx_and_dx_dr_8i(node_x, element):
    x=node_x[element]
    r=get_integration_point_8i(dtype=node_x.dtype, device=node_x.device)
    d_sf_dx, dx_dr, det_dx_dr = cal_d_sf_dh(r, x)
    return d_sf_dx, dx_dr, det_dx_dr
#%%
def cal_d_sf_dX_and_dX_dr_1i(node_X, element):
    X=node_X[element]
    r=get_integration_point_1i(dtype=node_X.dtype, device=node_X.device)
    d_sf_dX, dX_dr, det_dX_dr = cal_d_sf_dh(r, X)
    return d_sf_dX, dX_dr, det_dX_dr
#%%
def cal_d_sf_dX_and_dX_dr_8i(node_X, element):
    X=node_X[element]
    r=get_integration_point_8i(dtype=node_X.dtype, device=node_X.device)
    d_sf_dX, dX_dr, det_dX_dr=cal_d_sf_dh(r, X)
    return d_sf_dX, dX_dr, det_dX_dr
#%% forward: node_x is unknown, node_X is known
def cal_cauchy_stress_force(node_x, element, d_sf_dX, material, cal_cauchy_stress,
                            F0=None, return_F_S_W=False, return_stiffness=None, return_force_of_element=False):
    x=node_x[element].requires_grad_(True)
    if d_sf_dX.shape[1] != 1 and d_sf_dX.shape[1] != 8:
        #shape should be (M,1,3,8) or (M,8,3,8)
        raise ValueError("d_sf_dX.shape[1] != 1 and d_sf_dX.shape[1] != 8")
    F=cal_F_tensor_with_d_sf_dX(x, d_sf_dX)
    if F0 is not None:
        #F0 is residual deformation
        F=matmul(F, F0)
    #-----------------------------------------------------------------------------
    #F.shape: (M,1,3,3) or (M,8,3,3)
    #M=element.shape[0]
    #material.shape: (1, A) or (M, A)
    S,W=cal_cauchy_stress(F, material, create_graph=True, return_W=True)
    #S.shape: (M,1,3,3) or (M,8,3,3)
    if d_sf_dX.shape[1] == 1:
        r=get_integration_point_1i(dtype=node_x.dtype, device=node_x.device)
    else:
        r=get_integration_point_8i(dtype=node_x.dtype, device=node_x.device)
    d_sf_dx, dx_dr, det_dx_dr=cal_d_sf_dh(r, x)
    force_element=cal_nodal_force_from_cauchy_stress(S, d_sf_dx, det_dx_dr)
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
    #out=[force, F, S, W, H, force_element]
    out=[force]
    if return_F_S_W == True:
        out.extend([F,S,W])
    if return_stiffness is not None:
        out.append(H)
    if return_force_of_element == True:
        out.append(force_element)
    return out
#%% node_x is known, node_X or material is unknown
def cal_cauchy_stress_force_inverse(d_sf_dx, det_dx_dr, n_nodes, element, F, material, cal_cauchy_stress,
                                    F0=None, return_S_W=False):
    #1i if d_sf_dx, det_dx_dr from cal_d_sf_dx_and_dx_dr_1i
    #8i if d_sf_dx, det_dx_dr from cal_d_sf_dx_and_dx_dr_8i
    if F0 is not None:
        #F0 is residual deformation
        F=matmul(F, F0)
    #-----------------------------------------------------------------------------
    S, W=cal_cauchy_stress(F, material, create_graph=True, return_W=True)
    #S.shape: (M,8,3,3)
    force_element=cal_nodal_force_from_cauchy_stress(S, d_sf_dx, det_dx_dr)
    #force_element.shape: (M,8,3)
    #-----------------------------------------------------------------------------
    #M=element.shape[0]
    force = torch_scatter.scatter(force_element.view(-1,3), element.view(-1), dim=0, dim_size=n_nodes, reduce="sum")
    #force.shape: (n_nodes, 3)
    #-----------------------------------------------------------------------------
    if return_S_W == False:
        return force
    else:
        return force, S, W
#%%
def cal_1pk_stress_force(node_x, element, d_sf_dX, det_dX_dr, material, cal_1pk_stress,
                         F0=None, return_F_S_W=False, return_stiffness=None, return_force_of_element=False):
    x=node_x[element].requires_grad_(True)
    if d_sf_dX.shape[1] != 1 and d_sf_dX.shape[1] != 8:
        #shape should be (M,1,3,8) or (M,8,3,8)
        raise ValueError("d_sf_dX.shape[1] != 1 and d_sf_dX.shape[1] != 8")
    F=cal_F_tensor_with_d_sf_dX(x, d_sf_dX)
    if F0 is not None:
        #F0 is residual deformation
        F=matmul(F, F0)
    #F.shape: (M,1,3,3) or (M,8,3,3)
    #material.shape: (M,A) or (1,A)
    S, W=cal_1pk_stress(F, material, create_graph=True, return_W=True)
    #S.shape: (M,1,3,3) or (M,8,3,3)
    force_element=cal_nodal_force_from_1pk_stress(S, d_sf_dX, det_dX_dr)
    #force_element.shape: (M,8,3)
    #force_external.shape: (N,3)
    #-----------------------------------------------------------------------------
    N=node_x.shape[0]
    #M=element.shape[0]
    force = torch_scatter.scatter(force_element.view(-1,3), element.view(-1), dim=0, dim_size=N, reduce="sum")
    #force_internal.shape: (N, 3)
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
    #out=[force, F, S, W, H, force_element]
    out=[force]
    if return_F_S_W == True:
        out.extend([F,S,W])
    if return_stiffness is not None:
        out.append(H)
    if return_force_of_element == True:
        out.append(force_element)
    return out
#%%
def cal_diagonal_hessian(loss_fn, node_x):
    node_x=node_x.detach()
    N=node_x.shape[0]
    x=[]
    for n in range(0, N):
        x.append(node_x[n].view(1,3).requires_grad_(True))
    node=torch.cat(x, dim=0)
    loss=loss_fn(node)
    Hdiag=torch.zeros((3*node_x.shape[0],), dtype=node_x.dtype, device=node_x.device)
    g_all=torch.autograd.grad(loss, x, create_graph=True)
    for n in range(0, N):
        g=g_all[n]
        gg0=torch.autograd.grad(g[0,0], x[n], retain_graph=True)[0]
        gg1=torch.autograd.grad(g[0,1], x[n], retain_graph=True)[0]
        gg2=torch.autograd.grad(g[0,2], x[n], retain_graph=True)[0]
        Hdiag[3*n]=gg0[0,0]
        Hdiag[3*n+1]=gg1[0,1]
        Hdiag[3*n+2]=gg2[0,2]
    Hdiag=Hdiag.detach()
    return Hdiag
#%%
def cal_attribute_on_node(n_nodes, element, element_attribute):
    #n_nodes: the number of nodes
    #element.shape (M,8)
    #element_attribute can be stress, F tensor, detF, shape is (M,?,?,...)
    a_shape=list(element_attribute.shape)
    M=a_shape[0]
    a=1
    if len(a_shape) > 1:
        a=np.prod(a_shape[1:])
    attribute=element_attribute.view(M,1,a).expand(M,8,a).contiguous()
    attribute=torch_scatter.scatter(attribute.view(-1,a), element.view(-1), dim=0, dim_size=n_nodes, reduce="mean")
    a_shape[0]=n_nodes
    attribute=attribute.view(a_shape)
    return attribute
#%%
def cal_cauchy_stress_from_1pk_stress(S, F):
    #S, F from run_net_1pk
    J=torch.det(F).view(F.shape[0],F.shape[1],1,1)
    Ft=F.permute(0,1,3,2)
    Sigma=(1/J)*torch.matmul(S, Ft)
    return Sigma

