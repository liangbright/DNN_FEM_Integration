#%% quad element of a 3D surface
#     v
#    /|\
#     |
# x3  |  x2
#     0--------->u
# x0     x1
# r=[u,v] is the locaiton of a point in the unit element, u=r[0], v=r[1]
# r(x0)=[-1,-1], r(x1)=[1,-1], r(x2)=[1,1], r(x3)=[-1,1]
# integration point location:
#     https://www.mm.bme.hu/~gyebro/files/ans_help_v182/ans_thry/thy_et1.html
#%%
import torch
from torch import cross
from math import sqrt
#%%
def get_integration_point_1i():
    r=[0, 0]
    return r
#%%
def get_integration_point_4i():
    a=1/sqrt(3) #0.577350269189626
    r0=[-a,-a] # the location of the integration point #1
    r1=[+a,-a] # the location of the integration point #2
    r2=[+a,+a] # the location of the integration point #3
    r3=[-a,+a] # the location of the integration point #4
    return r0, r1, r2, r3
#%%
def sf0(r):
    #r is [u, v], an integration point
    return (1/4)*(1-r[0])*(1-r[1])
#%%
def sf1(r):
    #r is [u, v], an integration point
    return (1/4)*(1+r[0])*(1-r[1])
#%%
def sf2(r):
    #r is [u, v], an integration point
    return (1/4)*(1+r[0])*(1+r[1])
#%%
def sf3(r):
    #r is [u, v], an integration point
    return (1/4)*(1-r[0])*(1+r[1])
#%%
def get_weight_4i():
    r0, r1, r2, r3= get_integration_point_4i()
    #wij: shape_function_i_at_integration_point_j * integration_weight_at_j
    #integration_weight_at_j is 1 for j=0,1,2,3
    w00=sf0(r0); w01=sf0(r1); w02=sf0(r2); w03=sf0(r3)
    #-----------------------
    w10=sf1(r0); w11=sf1(r1); w12=sf1(r2); w13=sf1(r3)
    #-----------------------
    w20=sf2(r0); w21=sf2(r1); w22=sf2(r2); w23=sf2(r3)
    #-----------------------
    w30=sf3(r0); w31=sf3(r1); w32=sf3(r2); w33=sf3(r3)
    #-----------------------
    return (w00, w01, w02, w03,
            w10, w11, w12, w13,
            w20, w21, w22, w23,
            w30, w31, w32, w33)
#%%
def interpolate(r, h0, h1, h2, h3):
    #r is [u, v], an integration point
    #h0.shape is (M,3)
    h=(1/4)*((1-r[0])*(1-r[1])*h0
            +(1+r[0])*(1-r[1])*h1
            +(1+r[0])*(1+r[1])*h2
            +(1-r[0])*(1+r[1])*h3)
    return h
#%%
def cal_dh_du(r, h0, h1, h2, h3):
    #r is [u, v], an integration point
    #h0.shape is (M,3)
    dh_du=(1/4)*(-(1-r[1])*h0
                 +(1-r[1])*h1
                 +(1+r[1])*h2
                 -(1+r[1])*h3)
    return dh_du #(M,3)
#%%
def cal_dh_dv(r, h0, h1, h2, h3):
    #r is [u, v], an integration point
    #h0.shape is (M,3)
    dh_dv=(1/4)*(-(1-r[0])*h0
                 -(1+r[0])*h1
                 +(1+r[0])*h2
                 +(1-r[0])*h3)
    return dh_dv #(M,3)
#%%
def cal_dh_dr(r, h0, h1, h2, h3, numerator_layout=True):
    #r is [u, v], an integration point
    #h0.shape is (M,3), it is x, X, or u=x-X
    #dh_dr: (numerator_layout is True)
    # h=[a,b,c]
    # da/du, da/dv
    # db/du, db/dv
    # dc/du, dc/dv
    #----------------------------------
    M=h0.shape[0]
    dh_du=cal_dh_du(r, h0, h1, h2, h3).view(M,3,1)
    dh_dv=cal_dh_dv(r, h0, h1, h2, h3).view(M,3,1)
    dh_dr=torch.cat([dh_du, dh_dv], dim=-1)#(M,3,2)
    if numerator_layout == False:
        dh_dr=dh_dr.permute(0,2,1)
    return dh_dr
#%%
def cal_normal(r, x0, x1, x2, x3):
    #r is [u, v], an integration point
    #x0.shape is (M,3)
    dx_du=cal_dh_du(r, x0, x1, x2, x3)
    dx_dv=cal_dh_dv(r, x0, x1, x2, x3)
    normal=cross(dx_du, dx_dv)
    normal=normal/torch.norm(normal, p=2, dim=1, keepdim=True)
    return normal #(M,3)
#%%
if __name__ == "__main__":
    #%%
    device_gpu=torch.device("cuda:1")
    device_cpu=torch.device("cpu")
    x=torch.rand(1, 4, 3, device=device_gpu)
    import time
    t0=time.time()
    dh_du=cal_dh_du([0,0], x[:,0], x[:,1], x[:,2], x[:,3])
    dh_dv=cal_dh_dv([0,0], x[:,0], x[:,1], x[:,2], x[:,3])
    dh_dr=cal_dh_dr([0,0], x[:,0], x[:,1], x[:,2], x[:,3])
    normal=cal_normal([0,0], x[:,0], x[:,1], x[:,2], x[:,3])
    t1=time.time()
    print('time cost', t1-t0)

