# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 20:59:45 2021

@author: liang
"""
import torch
import numpy
def cal_von_mises_stress(S, apply_sqrt=True):
    #S is cauchy stress
    Sxx=S[...,0,0]
    Syy=S[...,1,1]
    Szz=S[...,2,2]
    Sxy=S[...,0,1]
    Syz=S[...,1,2]
    Szx=S[...,2,0]
    VM=Sxx**2+Syy**2+Szz**2-Sxx*Syy-Syy*Szz-Szz*Sxx+3*(Sxy**2+Syz**2+Szx**2)
    VM[VM<0]=0
    if apply_sqrt == True:
        if isinstance(S, torch.Tensor):
            VM=torch.sqrt(VM)
        elif isinstance(S, numpy.ndarray):
            VM=numpy.sqrt(VM)
        else:
            raise ValueError("unkown type(S):"+str(type(S)))
    return VM
