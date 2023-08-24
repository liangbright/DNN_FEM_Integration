# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 00:41:30 2021

@author: liang
"""
import torch
from torch import cross
from Element_Quad_3D import cal_dh_du, cal_dh_dv, get_integration_point_1i, get_integration_point_4i, get_weight_4i
#%% quad surface of a deformed C3D8 element
# x3    x2
#
# x0    x1
# r=[u,v] is the locaiton of a point in the unit element
# integration point location
#     https://www.mm.bme.hu/~gyebro/files/ans_help_v182/ans_thry/thy_et1.html
#%% use one integration point
def cal_nodal_force_from_pressure_1i(pressure, x):
    #x.shape (M,4,3)
    #pressure is scalar or tensor with shape (M, 1)
    x0=x[:,0]; x1=x[:,1]; x2=x[:,2]; x3=x[:,3]
    r=get_integration_point_1i()
    dx_du=cal_dh_du(r, x0, x1, x2, x3)
    dx_dv=cal_dh_dv(r, x0, x1, x2, x3)
    #print(dx_du.shape, dx_dv.shape)
    #force_k=Integration(pressure*cross(dx_du, dx_dv)*shape_function_k at r)
    #w=(1/4)*(2**2)=1 # shape_function_weight * integration_weight = 1
    force=pressure*cross(dx_du, dx_dv)
    M=x0.shape[0]# the number of elements
    force=force.view(M, 1, 3)
    force=force.expand(M, 4, 3).contiguous()
    return force
#%% use four integration points
def cal_nodal_force_from_pressure_4i(pressure, x):
    #x.shape (M,4,3)
    #pressure is scalar or tensor with shape (M, 1)
    x0=x[:,0]; x1=x[:,1]; x2=x[:,2]; x3=x[:,3]
    r0, r1, r2, r3=get_integration_point_4i()
    (w00, w01, w02, w03,
     w10, w11, w12, w13,
     w20, w21, w22, w23,
     w30, w31, w32, w33)=get_weight_4i()
    dx_du_r0=cal_dh_du(r0, x0, x1, x2, x3); dx_dv_r0=cal_dh_dv(r0, x0, x1, x2, x3)
    dx_du_r1=cal_dh_du(r1, x0, x1, x2, x3); dx_dv_r1=cal_dh_dv(r1, x0, x1, x2, x3)
    dx_du_r2=cal_dh_du(r2, x0, x1, x2, x3); dx_dv_r2=cal_dh_dv(r2, x0, x1, x2, x3)
    dx_du_r3=cal_dh_du(r3, x0, x1, x2, x3); dx_dv_r3=cal_dh_dv(r3, x0, x1, x2, x3)
    #print(dx_du_r0.shape, dx_du_r1.shape, dx_du_r2.shape, dx_du_r3.shape)
    #force_k=Integration(pressure*cross(dx_du, dx_dv)*shape_function_k at r)
    c0=cross(dx_du_r0, dx_dv_r0)
    c1=cross(dx_du_r1, dx_dv_r1)
    c2=cross(dx_du_r2, dx_dv_r2)
    c3=cross(dx_du_r3, dx_dv_r3)
    force0=pressure*(w00*c0+w01*c1+w02*c2+w03*c3)
    force1=pressure*(w10*c0+w11*c1+w12*c2+w13*c3)
    force2=pressure*(w20*c0+w21*c1+w22*c2+w23*c3)
    force3=pressure*(w30*c0+w31*c1+w32*c2+w33*c3)
    M=x0.shape[0]# the number of elements
    force0=force0.view(M, 1, 3)
    force1=force1.view(M, 1, 3)
    force2=force2.view(M, 1, 3)
    force3=force3.view(M, 1, 3)
    force = torch.cat([force0, force1, force2, force3], dim=1)
    return force
#%%
if __name__ == "__main__":
    #%%
    device_gpu=torch.device("cuda:0")
    device_cpu=torch.device("cpu")
    torch.manual_seed(100)
    x=torch.rand(1, 4, 3, device=device_gpu)
    print("x", x)
    #%%
    import time
    pressure=torch.tensor(16.0, device=device_gpu)
    t0=time.time()
    force_1i = cal_nodal_force_from_pressure_1i(pressure, x)
    print("force_1i", force_1i)
    t1=time.time()
    print('time cost', t1-t0)
    t0=time.time()
    force_4i= cal_nodal_force_from_pressure_4i(pressure, x)
    print("force_4i", force_4i)
    t1=time.time()
    print('time cost', t1-t0)
