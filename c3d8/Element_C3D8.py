# -*- encoding: utf-8 -*-
"""
Fri march 26 22:51:21 2021

@author: liang
"""
#%%
#integration point location
#https://www.mm.bme.hu/~gyebro/files/ans_help_v182/ans_thry/thy_et1.html
#%%
import torch
from torch import matmul
from torch.linalg import inv, det, solve
from math import sqrt
#%%
def get_integration_point_1i(dtype, device):
    r=torch.tensor([[0,0,0]], dtype=dtype, device=device) #r.shape (1, 3)
    return r
#%%
def get_integration_point_8i(dtype, device):
    a=1/sqrt(3) #a=0.5773502691896258
    r0=[-a,-a,-a] # the location of the integration point #1
    r1=[+a,-a,-a] # the location of the integration point #2
    r2=[-a,+a,-a] # the location of the integration point #3
    r3=[+a,+a,-a] # the location of the integration point #4
    r4=[-a,-a,+a] # the location of the integration point #5
    r5=[+a,-a,+a] # the location of the integration point #6
    r6=[-a,+a,+a] # the location of the integration point #7
    r7=[+a,+a,+a] # the location of the integration point #8
    r=torch.tensor([r0, r1, r2, r3, r4, r5, r6, r7], dtype=dtype, device=device) #r.shape (8, 3)
    return r
#%% shape function 0
def sf0(r):
    return (1/8)*(1-r[0])*(1-r[1])*(1-r[2])
#%%
def cal_d_sf0_dr(r):
    dr0=-(1/8)*(1-r[1])*(1-r[2])
    dr1=-(1/8)*(1-r[0])*(1-r[2])
    dr2=-(1/8)*(1-r[0])*(1-r[1])
    dr=[dr0, dr1, dr2]
    return dr
#%%
def sf1(r):
    return (1/8)*(1+r[0])*(1-r[1])*(1-r[2])
#%%
def cal_d_sf1_dr(r):
    dr0=+(1/8)*(1-r[1])*(1-r[2])
    dr1=-(1/8)*(1+r[0])*(1-r[2])
    dr2=-(1/8)*(1+r[0])*(1-r[1])
    dr=[dr0, dr1, dr2]
    return dr
#%%
def sf2(r):
    return (1/8)*(1+r[0])*(1+r[1])*(1-r[2])
#%%
def cal_d_sf2_dr(r):
    dr0=+(1/8)*(1+r[1])*(1-r[2])
    dr1=+(1/8)*(1+r[0])*(1-r[2])
    dr2=-(1/8)*(1+r[0])*(1+r[1])
    dr=[dr0, dr1, dr2]
    return dr
#%%
def sf3(r):
    return (1/8)*(1-r[0])*(1+r[1])*(1-r[2])
#%%
def cal_d_sf3_dr(r):
    dr0=-(1/8)*(1+r[1])*(1-r[2])
    dr1=+(1/8)*(1-r[0])*(1-r[2])
    dr2=-(1/8)*(1-r[0])*(1+r[1])
    dr=[dr0, dr1, dr2]
    return dr
#%%
def sf4(r):
    return (1/8)*(1-r[0])*(1-r[1])*(1+r[2])
#%%
def cal_d_sf4_dr(r):
    dr0=-(1/8)*(1-r[1])*(1+r[2])
    dr1=-(1/8)*(1-r[0])*(1+r[2])
    dr2=+(1/8)*(1-r[0])*(1-r[1])
    dr=[dr0, dr1, dr2]
    return dr
#%%
def sf5(r):
    return (1/8)*(1+r[0])*(1-r[1])*(1+r[2])
#%%
def cal_d_sf5_dr(r):
    dr0=+(1/8)*(1-r[1])*(1+r[2])
    dr1=-(1/8)*(1+r[0])*(1+r[2])
    dr2=+(1/8)*(1+r[0])*(1-r[1])
    dr=[dr0, dr1, dr2]
    return dr
#%%
def sf6(r):
    return (1/8)*(1+r[0])*(1+r[1])*(1+r[2])
#%%
def cal_d_sf6_dr(r):
    dr0=+(1/8)*(1+r[1])*(1+r[2])
    dr1=+(1/8)*(1+r[0])*(1+r[2])
    dr2=+(1/8)*(1+r[0])*(1+r[1])
    dr=[dr0, dr1, dr2]
    return dr
#%%
def sf7(r):
    return (1/8)*(1-r[0])*(1+r[1])*(1+r[2])
#%%
def cal_d_sf7_dr(r):
    dr0=-(1/8)*(1+r[1])*(1+r[2])
    dr1=+(1/8)*(1-r[0])*(1+r[2])
    dr2=+(1/8)*(1-r[0])*(1+r[1])
    dr=[dr0, dr1, dr2]
    return dr
#%%
def cal_d_sf_dr(r):
    #r.shape is (K,3), K is the number of integration points
    K=r.shape[0]
    r=r.permute(1,0)#shape is (3,K)
    r=r.view(3,K,1,1)
    d_sf0_dr=torch.cat(cal_d_sf0_dr(r), dim=1)#shape is (K,3,1)
    d_sf1_dr=torch.cat(cal_d_sf1_dr(r), dim=1)#shape is (K,3,1)
    d_sf2_dr=torch.cat(cal_d_sf2_dr(r), dim=1)#shape is (K,3,1)
    d_sf3_dr=torch.cat(cal_d_sf3_dr(r), dim=1)#shape is (K,3,1)
    d_sf4_dr=torch.cat(cal_d_sf4_dr(r), dim=1)#shape is (K,3,1)
    d_sf5_dr=torch.cat(cal_d_sf5_dr(r), dim=1)#shape is (K,3,1)
    d_sf6_dr=torch.cat(cal_d_sf6_dr(r), dim=1)#shape is (K,3,1)
    d_sf7_dr=torch.cat(cal_d_sf7_dr(r), dim=1)#shape is (K,3,1)
    d_sf_dr=[d_sf0_dr, d_sf1_dr, d_sf2_dr, d_sf3_dr, d_sf4_dr, d_sf5_dr, d_sf6_dr, d_sf7_dr]
    d_sf_dr=torch.cat(d_sf_dr, dim=2)
    d_sf_dr=d_sf_dr.view(1,K,3,8)#8: sf0 to sf7
    return d_sf_dr
#%%
def interpolate(r, h):
    #r.shape (K,3), K is the number of integration points
    #h.shape (M,8,3), M elements, 1 elemnet has 8 nodes, 1 node has a 3D position, h is x or X or u
    #hr=sum_k{sf_k(r)*h_k, k=0 to 7}
    K=r.shape[0]
    r=r.permute(1,0)#shape is (3,K)
    r=r.view(3,K,1)
    w=torch.cat([sf0(r), sf1(r), sf2(r), sf3(r), sf4(r), sf5(r), sf6(r), sf7(r)], dim=1)
    w=w.view(1,K,8,1)
    M=h.shape[0]
    h=h.view(M,1,8,3)
    hr=(w*h).sum(dim=2)#shape (M,K,3)
    return hr
#%%
def cal_dh_dr(r, h, d_sf_dr=None, numerator_layout=True):
    if d_sf_dr is None:
        d_sf_dr=cal_d_sf_dr(r)
    #r.shape (K,3), K is the number of integration points
    #h.shape (M,8,3), M elements, 1 elemnet has 8 nodes, 1 node has a 3D position
    #d_sf_dr.shape (1,K,3,8)
    M=h.shape[0]
    h=h.view(M,1,8,3)
    dh_dr=matmul(d_sf_dr, h) #shape (M,K,3,3)
    if numerator_layout == True:
        dh_dr=dh_dr.transpose(2,3)
    return dh_dr
#%%
def cal_d_sf_dh(r, h):
    #h is x, X or u=x-X
    #(page264) in the book dNi/dX at r, Ni is the shape function of node-i
    #(9.15b) in the book dNi/dx at r, Ni is the shape function of node-i
    d_sf_dr=cal_d_sf_dr(r)
    #d_sf_dr.shape (1,K,3,8)
    dh_dr=cal_dh_dr(r, h, d_sf_dr, False)
    #dh_dr.shape is (M,K,3,3)
    det_dh_dr=det(dh_dr).view(dh_dr.shape[0],-1,1,1)#shape is (M,K,1,1)
    #-----------------------------------
    #inv_dh_dr=inv(dh_dr)
    #t_inv_dh_dr=inv_dh_dr.permute(0,1,3,2)# (9.6ab) transposed inverse
    #d_sf_dh=matmul(t_inv_dh_dr, d_sf_dr)
    #-----------------------------------
    #d_sf_dh=solve(dh_dr.transpose(2,3), d_sf_dr)
    d_sf_dh=solve(dh_dr, d_sf_dr) # numerator_layout=False for cal_dh_dr
    #d_sf_dh.shape is (M,K,3,8)
    #d_sf_dh[:,:,:,0] is d_sf0_dx
    #-----------------------------------
    return d_sf_dh, dh_dr, det_dh_dr
#%%
def cal_F_tensor_old(r, x, X):
    #r.shape (K,3), K is the number of integration points
    #x.shape is (M,8,3)
    #X.shape is (M,8,3)
    dx_dr=cal_dh_dr(r, x)
    dX_dr=cal_dh_dr(r, X)
    F=matmul(dx_dr, inv(dX_dr))
    #F.shape is (M,K,3,3)
    return F
#%%
def cal_F_tensor(r, x, X):
    #r.shape (K,3), K is the number of integration points
    #x.shape is (M,8,3)
    #X.shape is (M,8,3)
    dx_dr=cal_dh_dr(r, x)
    dX_dr=cal_dh_dr(r, X)
    #F=matmul(dx_dr, inv(dX_dr))
    F=solve(dX_dr.transpose(2,3), dx_dr.transpose(2,3)).transpose(2,3)
    #F.shape is (M,K,3,3)
    return F
#%%
def cal_F_tensor_with_d_sf_dX(x, d_sf_dX):
    #x.shape is (M,8,3)
    #d_sf_dX.shape is (M,K,3,8), from cal_d_sf_dh
    M=x.shape[0]
    K=d_sf_dX.shape[1]
    x=x.view(M,1,8,3,1).expand(M,K,8,3,1)
    d_sf_dX=d_sf_dX.transpose(2,3)#shape is (M,K,8,3)
    d_sf_dX=d_sf_dX.view(M,K,8,1,3)
    F=matmul(x, d_sf_dX).sum(dim=2)
    return F
#%%
def cal_F_tensor_X_cube(r, x, LX):
    #r.shape (K,3), K is the number of integration points
    #x.shape is (M,8,3)
    #special case: X is a cube with length=LX
    dx_dr=cal_dh_dr(r, x)
    dX_dr=0.5*LX
    if isinstance(dX_dr, torch.Tensor):
        dX_dr=dX_dr.view(dX_dr.shape[0],1,1,1)
    F=dx_dr/dX_dr
    return F
#%% use one integration point
def cal_nodal_force_from_cauchy_stress_1i(S, d_sf_dx, det_dx_dr):
    #r=get_integration_point_1i()
    #F=cal_F_tensor(...) on the integration point, F.shape is (M,1,3,3)
    #S=cal_Cauchy_stress(F, Mat) on the integration point, S.shape is (M,1,3,3)
    #d_sf_dx, dx_dr, det_dx_dr=cal_d_sf_dh(r, x)
    #d_sf_dx.shape is (M,1,3,8), det_dx_dr.shape is (M,1,1,1)
    #det_dx_dr=det_dx_dr.abs() #it should be > 0
    #----------------------------------------------------------------------
    #force_k=integration(S*dNk_dx at r), #(9.15b) in the book
    #shape_function_weight is not used for stress-force calculation
    #w=2**3=8 is the integration_weight: the same for the eight integration points
    #print("shape", S.shape, d_sf_dx.shape, det_dx_dr.shape)
    force=8*(matmul(S, d_sf_dx)*det_dx_dr)
    force=force.view(-1,3,8)
    force=force.permute(0,2,1)
    force=force.contiguous()
    return force
#%% use eight integration points
def cal_nodal_force_from_cauchy_stress_8i(S, d_sf_dx, det_dx_dr):
    #r=get_integration_point_8i(dtype, device)
    #F=cal_F_tensor(...) on the eight integration points, F.shape is (M,8,3,3)
    #S=cal_Cauchy_stress(F, Mat) the eight integration points， S.shape is (M,8,3,3)
    #d_sf_dx, dx_dr, det_dx_dr=cal_d_sf_dh(r, x)
    #d_sf_dx.shape is (M,8,3,8), det_dx_dr.shape is (M,8,1,1)
    #det_dx_dr=det_dx_dr.abs() #it should be > 0
    #----------------------------------------------------------------------
    #force_k=integration(S*dNk_dx at r), #(9.15b) in the book
    #shape_function_weight is not used for stress-force calculation
    #w=1 is the integration_weight: the same for the eight integration points
    force=(matmul(S, d_sf_dx)*det_dx_dr).sum(dim=1)#force.shape is (M,3,8)
    force=force.permute(0,2,1)
    force=force.contiguous()
    return force
#%%
def cal_nodal_force_from_cauchy_stress(S, d_sf_dx, det_dx_dr):
    if det_dx_dr.shape[1] == 1:
        return cal_nodal_force_from_cauchy_stress_1i(S, d_sf_dx, det_dx_dr)
    elif det_dx_dr.shape[1] == 8:
        return cal_nodal_force_from_cauchy_stress_8i(S, d_sf_dx, det_dx_dr)
    else:
        raise ValueError("only support 1i and 8i")
#%% use one integration point
def cal_nodal_force_from_2pk_stress_1i(F, S, d_sf_dX, det_dX_dr):
    #r=get_integration_point_1i()
    #F=cal_F_tensor(...) on the integration point, F.shape is (M,1,3,3)
    #S=cal_2pk_stress(F, Mat) on the integration point, S.shape is (M,1,3,3)
    #d_sf_dX, dX_dr, det_dX_dr=cal_d_sf_dh(r, X)
    #d_sf_dX.shape is (M,1,3,8), det_dX_dr.shape is (M,1,1,1)
    #det_dx_dr=det_dx_dr.abs() #it should be > 0
    #----------------------------------------------------------------
    #force_k=integration(F*S*dNi_dx at r), #(page264) in the book
    #shape_function_weight is not used for stress-force calculation
    #w=2**3=8 is the integration_weight: the same for the eight integration points
    force=8*(matmul(matmul(F, S), d_sf_dX)*det_dX_dr)
    force=force.view(-1,3,8)
    force=force.permute(0,2,1)
    force=force.contiguous()
    return force
#%% use eight integration points
def cal_nodal_force_from_2pk_stress_8i(F, S, d_sf_dX, det_dX_dr):
    #r=get_integration_point_8i(dtype, device)
    #F=cal_F_tensor(...) on the eight integration points, F.shape is (M,8,3,3)
    #S=cal_2pk_stress(F, Mat) on the eight integration points, S.shape is (M,8,3,3)
    #d_sf_dX, dX_dr, det_dX_dr=cal_d_sf_dh(r, X)
    #d_sf_dX.shape is (M,8,3,8), det_dX_dr.shape is (M,8,1,1)
    #det_dx_dr=det_dx_dr.abs() #it should be > 0
    #---------------------------------------------------------------
    #force_i=integration(F*S*dNi_dx at r), #(9.15b) in the book
    #shape_function_weight is not used for stress-force calculation
    #w=1 is the integration_weight: the same for the eight integration points
    force=(matmul(matmul(F, S), d_sf_dX)*det_dX_dr).sum(dim=1)
    force=force.permute(0,2,1)
    force=force.contiguous()
    return force
#%%
def cal_nodal_force_from_2pk_stress(F, S, d_sf_dX, det_dX_dr):
    if det_dX_dr.shape[1] == 1:
        return cal_nodal_force_from_2pk_stress_1i(F, S, d_sf_dX, det_dX_dr)
    elif det_dX_dr.shape[1] == 8:
        return cal_nodal_force_from_2pk_stress_8i(F, S, d_sf_dX, det_dX_dr)
    else:
        raise ValueError("only support 1i and 8i")
#%% use one integration point
def cal_nodal_force_from_1pk_stress_1i(S, d_sf_dX, det_dX_dr):
    #r=get_integration_point_1i()
    #F=cal_F_tensor(...) on the integration point, F.shape is (M,1,3,3)
    #S=cal_1pk_stress(F, Mat) on the integration point, S.shape is (M,1,3,3)
    #d_sf_dX, dX_dr, det_dX_dr=cal_d_sf_dh(r, X)
    #d_sf_dX.shape is (M,1,3,8), det_dX_dr.shape is (M,1,1,1)
    #det_dx_dr=det_dx_dr.abs() #it should be > 0
    #----------------------------------------------------------------
    #force_k=integration(F*S*dNi_dx at r), #(page264) in the book
    #S_1pk=F*S_2pk
    #shape_function_weight is not used for stress-force calculation
    #w=2**3=8 is the integration_weight: the same for the eight integration points
    force=8*(matmul(S, d_sf_dX)*det_dX_dr)
    force=force.view(-1,3,8)
    force=force.permute(0,2,1)
    force=force.contiguous()
    return force
#%% use eight integration points
def cal_nodal_force_from_1pk_stress_8i(S, d_sf_dX, det_dX_dr):
    #r=get_integration_point_8i(dtype, device)
    #F=cal_F_tensor(...) on the eight integration points, F.shape is (M,8,3,3)
    #S=cal_1pk_stress(F, Mat) on the eight integration points, S.shape is (M,8,3,3)
    #d_sf_dX, dX_dr, det_dX_dr=cal_d_sf_dh(r, X)
    #d_sf_dX.shape is (M,8,3,8), det_dX_dr.shape is (M,8,1,1)
    #det_dx_dr=det_dx_dr.abs() #it should be > 0
    #---------------------------------------------------------------
    #force_i=integration(F*S*dNi_dx at r), #(9.15b) in the book
    #S_1pk=F*S_2pk
    #shape_function_weight is not used for stress-force calculation
    #w=1 is the integration_weight: the same for the eight integration points
    force=(matmul(S, d_sf_dX)*det_dX_dr).sum(dim=1)
    force=force.permute(0,2,1)
    force=force.contiguous()
    return force
#%%
def cal_nodal_force_from_1pk_stress(S, d_sf_dX, det_dX_dr):
    if det_dX_dr.shape[1] == 1:
        return cal_nodal_force_from_1pk_stress_1i(S, d_sf_dX, det_dX_dr)
    elif det_dX_dr.shape[1] == 8:
        return cal_nodal_force_from_1pk_stress_8i(S, d_sf_dX, det_dX_dr)
    else:
        raise ValueError("only support 1i and 8i")
#%% use one integration point to calculate the strain energy of each element
def cal_strain_energy_1i(W, det_dX_dr, reduction):
    #W.shape (M,1)
    #det_dX_dr.shape is (M,1,1,1)
    #r=[0, 0, 0] #the location of the integration point
    #W=cal_strain_energy_density(F, Mat) on the integration point
    #energy=integration(W at r)
    #integration weight is 2**3
    det_dX_dr=det_dX_dr.view(W.shape)
    energy=8*(W*det_dX_dr)
    #shape is (M,1)
    if reduction is None or reduction == "none":
        pass
    elif reduction == "sum":
        energy=energy.sum()
    elif reduction == "mean":
        energy=energy.mean()
    return energy
#%% use eight integration points to calculate the strain energy of each element
def cal_strain_energy_8i(W, det_dX_dr, reduction):
    #W.shape (M,8)
    #det_dX_dr.shape (M,8,1,1)
    #W=cal_strain_energy_density(F, Mat) on the 8 integration points
    #energy=integration(W at r)
    det_dX_dr=det_dX_dr.view(W.shape)
    energy=(W*det_dX_dr).sum(dim=1, keepdim=True)
    #shape is (M,1)
    if reduction is None or reduction == "none":
        pass
    elif reduction == "sum":
        energy=energy.sum()
    elif reduction == "mean":
        energy=energy.mean()
    return energy
#%%
def cal_strain_energy(W, det_dX_dr, reduction):
    if det_dX_dr.shape[1]==1:
        return cal_strain_energy_1i(W, det_dX_dr, reduction)
    elif det_dX_dr.shape[1]==8:
        return cal_strain_energy_8i(W, det_dX_dr, reduction)
    else:
        raise ValueError("only support 1i and 8i, det_dX_dr.shape[1]="+str(det_dX_dr.shape[1]))
#%% useful for 9.15c in the book
def get_weight_8i():
    r0, r1, r2, r3, r4, r5, r6, r7=get_integration_point_8i()
    #wij: shape_function_i_at_integration_point_j * integration_weight_at_j
    #integration_weight_at_j is 1 for j=0,1,2,3,4,5,6,7
    w00=sf0(r0); w01=sf0(r1); w02=sf0(r2); w03=sf0(r3)
    w04=sf0(r4); w05=sf0(r5); w06=sf0(r6); w07=sf0(r7)
    #----------------------
    w10=sf1(r0); w11=sf1(r1); w12=sf1(r2); w13=sf1(r3)
    w14=sf1(r4); w15=sf1(r5); w16=sf1(r6); w17=sf1(r7)
    #----------------------
    w20=sf2(r0); w21=sf2(r1); w22=sf2(r2); w23=sf2(r3)
    w24=sf2(r4); w25=sf2(r5); w26=sf2(r6); w27=sf2(r7)
    #----------------------
    w30=sf3(r0); w31=sf3(r1); w32=sf3(r2); w33=sf3(r3)
    w34=sf3(r4); w35=sf3(r5); w36=sf3(r6); w37=sf3(r7)
    #----------------------
    w40=sf4(r0); w41=sf4(r1); w42=sf4(r2); w43=sf4(r3)
    w44=sf4(r4); w45=sf4(r5); w46=sf4(r6); w47=sf4(r7)
    #----------------------
    w50=sf5(r0); w51=sf5(r1); w52=sf5(r2); w53=sf5(r3)
    w54=sf5(r4); w55=sf5(r5); w56=sf5(r6); w57=sf5(r7)
    #----------------------
    w60=sf6(r0); w61=sf6(r1); w62=sf6(r2); w63=sf6(r3)
    w64=sf6(r4); w65=sf6(r5); w66=sf6(r6); w67=sf6(r7)
    #----------------------
    w70=sf7(r0); w71=sf7(r1); w72=sf7(r2); w73=sf7(r3)
    w74=sf7(r4); w75=sf7(r5); w76=sf7(r6); w77=sf7(r7)
    #----------------------
    return (w00, w01, w02, w03, w04, w05, w06, w07,
            w10, w11, w12, w13, w14, w15, w16, w17,
            w20, w21, w22, w23, w24, w25, w26, w27,
            w30, w31, w32, w33, w34, w35, w36, w37,
            w40, w41, w42, w43, w44, w45, w46, w47,
            w50, w51, w52, w53, w54, w55, w56, w57,
            w60, w61, w62, w63, w64, w65, w66, w67,
            w70, w71, w72, w73, w74, w75, w76, w77)
#%%
if __name__ == "__main__":
    #%%
    device_gpu=torch.device("cuda:0")
    device_cpu=torch.device("cpu")
    #%%
    dh_dr0=torch.tensor([[1, 2, 3]])
    print("dh_dr0.shape", dh_dr0.shape)
    dh_dr1=torch.tensor([[4, 5, 6]])
    dh_dr2=torch.tensor([[7, 8, 9]])
    dh_dr=torch.cat([dh_dr0, dh_dr1, dh_dr2], dim=1)
    dh_dr=dh_dr.view(1,3,3)
    dh_dr=dh_dr.permute(0,2,1)
    print("dh_dr", dh_dr)
    print(dh_dr.reshape(-1))
    #%%
    dh_dr0=torch.tensor([[1, 2, 3]]).view(1,3,1)
    print("dh_dr0.shape", dh_dr0.shape)
    dh_dr1=torch.tensor([[4, 5, 6]]).view(1,3,1)
    dh_dr2=torch.tensor([[7, 8, 9]]).view(1,3,1)
    dh_dr=torch.cat([dh_dr0, dh_dr1, dh_dr2], dim=2)
    dh_dr=dh_dr.view(1,3,3)
    #dh_dr=dh_dr.permute(0,2,1)
    print("dh_dr", dh_dr)
    print(dh_dr.reshape(-1))
    #%%
    import time
    X=torch.rand(1, 8, 3).to(torch.float64)
    F=torch.rand(1, 3, 3).to(torch.float64)
    x=torch.matmul(F, X.view(1,8,3,1)).view(1,8,3)
    r=get_integration_point_1i(x.dtype, x.device)
    t0=time.time()
    F1=cal_F_tensor_old(r, x, X)
    t1=time.time()
    F2=cal_F_tensor(r,  x, X)
    t2=time.time()
    d_sf_dX, dX_d, det_dX_dr=cal_d_sf_dh(r, X)
    t3=time.time()
    F3=cal_F_tensor_with_d_sf_dX(x, d_sf_dX)
    t4=time.time()

    print('time cost1', t1-t0)
    print('time cost2', t2-t1)
    print('time cost3', t3-t2)
    print('time cost4', t4-t3)
    print((F2-F).abs().mean().item())
    print((F1-F2).abs().mean().item())
    print((F1-F2).abs().max().item())
    print((F1-F3).abs().mean().item())
    print((F1-F3).abs().max().item())
    print((F2-F3).abs().mean().item())
    print((F2-F3).abs().max().item())
    #%%
    x=torch.rand(64*64*64, 8, 3)
    X=torch.rand(64*64*64, 8, 3)
    r=get_integration_point_1i(x.dtype, x.device)
    t0=time.time()
    F=cal_F_tensor_X_cube(r, x, 1)
    t1=time.time()
    print('time cost3', t1-t0)
    #%%
    x=torch.rand(10000, 8, 3, device=device_gpu)
    S=torch.rand(10000, 1, 3, 3, device=device_gpu)
    t0=time.time()
    r=get_integration_point_1i(x.dtype, x.device)
    d_sf_dx, dx_dr, det_dx_dr=cal_d_sf_dh(r, x)
    force_1i=cal_nodal_force_from_cauchy_stress_1i(S, d_sf_dx, det_dx_dr)
    t1=time.time()
    print('time cost4', t1-t0)
    #%%
    x=torch.rand(10000, 8, 3, device=device_gpu)
    S=torch.rand(10000, 8, 3, 3, device=device_gpu)
    t0=time.time()
    r=get_integration_point_8i(dtype=x.dtype, device=x.device)
    d_sf_dx, dx_dr, det_dx_dr=cal_d_sf_dh(r, x)
    force_8i=cal_nodal_force_from_cauchy_stress_8i(S, d_sf_dx, det_dx_dr)
    t1=time.time()
    print('time cost5', t1-t0)
