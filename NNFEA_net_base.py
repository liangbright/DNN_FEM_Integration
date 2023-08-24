import torch
import torch.nn as nn
from torch.nn.functional import softplus, relu, elu
import numpy as np
#%%
class LinearRes(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super().__init__()
        self.in_dim=in_dim
        self.out_dim=out_dim
        self.activation=activation
        self.process_x = nn.Identity();
        if out_dim != in_dim:
            self.process_x = nn.Linear(in_dim, out_dim);
        self.linear1=nn.Linear(in_dim, out_dim)
        self.linear2=nn.Linear(out_dim, out_dim)
        if activation == 'softplus':
            self.activation=softplus
        elif activation == 'relu':
            self.activation=relu
        elif activation == "elu":
            self.activation=elu
        else:
            raise ValueError("unknown activation:"+activation)
    def forward(self, x):
        y_old=self.process_x(x)
        y_new=self.linear2(self.activation(self.linear1(x)))
        y=self.activation(y_old+y_new)
        return y
#%%
def init_weight(linear, alpha, is_first):
    in_dim=linear.in_features
    if is_first == True:
        linear.weight.data.uniform_(-1/in_dim, 1/in_dim)
    else:
        linear.weight.data.uniform_(-np.sqrt(6/in_dim)/alpha, np.sqrt(6/in_dim)/alpha)
#%%
class Sin(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, is_first):
        super().__init__()
        self.in_dim=in_dim
        self.out_dim=out_dim
        self.alpha=alpha
        self.is_first=is_first
        self.linear1=nn.Linear(in_dim, out_dim)
        init_weight(self.linear1, alpha, is_first)
    def forward(self, x):
        x1=self.linear1(self.alpha*x)
        y=torch.sin(x1)
        return y
#%%
class SinRes_old(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, is_first, out_activation):
        super().__init__()
        self.in_dim=in_dim
        self.out_dim=out_dim
        self.alpha=alpha
        self.is_first=is_first
        self.process_x = nn.Identity()
        if out_dim != in_dim:
            self.process_x = nn.Linear(in_dim, out_dim)
        #self.process_x1 is useful even if out_activation is None
        #it can adjust the range/std of the input to sin/cos, leading to faster convergence
        self.process_x1=nn.Linear(in_dim, out_dim, bias=False)
        init_weight(self.process_x1, alpha, is_first)
        self.linear2s=nn.Linear(out_dim, out_dim)
        if out_activation is None or out_activation == "none":
            self.out= nn.Identity()
        elif out_activation == 'softplus':
            self.out=nn.Softplus()
        elif out_activation == 'relu':
            self.out=nn.ReLU()
        elif out_activation == "elu":
            self.out=nn.ELU()
        else:
            raise ValueError("unknown out_activation:"+out_activation)
    def forward(self, x):
        y_old=self.process_x(x)
        x1=self.process_x1(self.alpha*x)
        y_new=self.linear2s(torch.sin(x1))
        y=self.out(y_old+y_new)
        return y
#%%
class SinRes(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, is_first):
        super().__init__()
        self.in_dim=in_dim
        self.out_dim=out_dim
        self.alpha=alpha
        self.is_first=is_first
        self.process_x = nn.Identity()
        if out_dim != in_dim:
            self.process_x = nn.Linear(in_dim, out_dim)
        self.linear=nn.Linear(in_dim, out_dim)
        init_weight(self.linear, alpha, is_first)
        self.c=0.5**0.5
    def forward(self, x):
        y_old=self.process_x(x)
        x1=self.linear(self.alpha*x)
        y_new=torch.sin(x1)
        y=self.c*(y_old+y_new)
        return y
#%%
class SinCos(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, is_first):
        super().__init__()
        self.in_dim=in_dim
        self.out_dim=out_dim
        self.alpha=alpha
        self.is_first=is_first
        self.linear1=nn.Linear(in_dim, out_dim//2)
        init_weight(self.linear1, alpha, is_first)
    def forward(self, x):
        x1=self.linear1(self.alpha*x)
        y=torch.cat([torch.sin(x1), torch.cos(x1)], dim=-1)
        return y
#%%
class SinCosRes(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, is_first, out_activation):
        super().__init__()
        self.in_dim=in_dim
        self.out_dim=out_dim
        self.alpha=alpha
        self.is_first=is_first
        self.process_x = nn.Identity()
        if out_dim != in_dim:
            self.process_x = nn.Linear(in_dim, out_dim)
        #self.process_x1 is useful even if out_activation is None
        #it can adjust the range/std of the input to sin/cos, leading to faster convergence
        self.process_x1=nn.Linear(in_dim, out_dim, bias=False)
        init_weight(self.process_x1, alpha, is_first)
        self.linear2s=nn.Linear(out_dim, out_dim, bias=False)
        self.linear2c=nn.Linear(out_dim, out_dim, bias=False)
        if out_activation is None or out_activation == "none":
            self.out= nn.Identity()
        elif out_activation == 'softplus':
            self.out=nn.Softplus()
        elif out_activation == 'relu':
            self.out=nn.ReLU()
        elif out_activation == "elu":
            self.out=nn.ELU()
        else:
            raise ValueError("unknown out_activation:"+out_activation)
    def forward(self, x):
        y_old=self.process_x(x)
        x1=self.process_x1(self.alpha*x)
        y_new=self.linear2s(torch.sin(x1))+self.linear2c(torch.cos(x1))
        y=self.out(y_old+y_new)
        return y
#%%
def preprocess(x, scale):
    if scale == 1:
        return x
    if isinstance(scale, tuple) or isinstance(scale, list):
        x_dim=x.shape[0]
        if len(scale) != x_dim:
            raise ValueError("len(scale) != x_dim")
        scale=torch.tensor(scale, dtype=x.dtype, device=x.device)
        scale=scale.view(1, x_dim)
        x=x*scale
    else:
        x=x*scale
    return x
#%%
class BaseNet0(nn.Module):
    def __init__(self, x_dim, h_dim, n_layers, x_scale, alpha, gamma, y_dim, activation="softplus", layer_norm=False):
        super().__init__()
        self.x_dim=x_dim
        self.h_dim=h_dim
        self.n_layers=n_layers
        self.x_scale=x_scale
        self.alpha=None
        self.gamma=gamma
        self.y_dim=y_dim
        self.activation=activation
        if activation == 'softplus':
            self.layer0=nn.Sequential(nn.Linear(x_dim, h_dim),
                                      nn.Softplus())
        elif activation == 'relu':
            self.layer0=nn.Sequential(nn.Linear(x_dim, h_dim),
                                      nn.ReLU())
        elif activation == 'elu':
            self.layer0=nn.Sequential(nn.Linear(x_dim, h_dim),
                                      nn.ELU())
        else:
            raise ValueError("error")
        layer1=[]
        for n in range(0, n_layers):
            layer1.append(nn.Linear(h_dim, h_dim))
            if activation == 'softplus':
                layer1.append(nn.Softplus())
            elif activation == 'relu':
                layer1.append(nn.ReLU())
            elif activation == "elu":
                layer1.append(nn.ELU())
            else:
                raise ValueError("unknown activation:"+activation)
            if layer_norm==True:
                layer1.append(nn.GroupNorm(1, h_dim))
        self.layer1=nn.Sequential(*layer1)
        self.layer2=nn.Linear(h_dim, y_dim)
        self.layer2.weight.data*=gamma
        self.layer2.bias.data*=0

    def forward(self, x):
        x=preprocess(x, self.x_scale)
        x0=self.layer0(x)
        x1=self.layer1(x0)
        y=self.layer2(x1)
        return y
#%%
class BaseNet1(nn.Module):
    def __init__(self, x_dim, h_dim, n_layers, x_scale, alpha, gamma, y_dim, activation="softplus", layer_norm=False):
        super().__init__()
        self.x_dim=x_dim
        self.h_dim=h_dim
        self.n_layers=n_layers
        self.x_scale=x_scale
        self.alpha=None
        self.gamma=gamma
        self.y_dim=y_dim
        self.activation=activation
        self.layer0=LinearRes(x_dim, h_dim, activation)
        layer1=[]
        for n in range(0, n_layers):
            layer1.append(LinearRes(h_dim, h_dim, activation))
            if layer_norm==True:
                layer1.append(nn.GroupNorm(1, h_dim))
        self.layer1=nn.Sequential(*layer1)
        self.layer2=nn.Linear(h_dim, y_dim)
        self.layer2.weight.data*=gamma
        self.layer2.bias.data*=0

    def forward(self, x):
        x=preprocess(x, self.x_scale)
        x0=self.layer0(x)
        x1=self.layer1(x0)
        y=self.layer2(x1)
        return y
#%%
class BaseNet1a(nn.Module):
    def __init__(self, x_dim, h_dim, n_layers, x_scale, alpha, gamma, y_dim, activation="softplus"):
        super().__init__()
        self.x_dim=x_dim
        self.h_dim=h_dim
        self.n_layers=n_layers
        self.x_scale=x_scale
        self.alpha=alpha
        self.gamma=gamma
        self.y_dim=y_dim
        self.activation=activation
        self.layer0=LinearRes(x_dim, h_dim, activation)
        layer1a=[]
        for n in range(0, n_layers):
            layer1a.append(LinearRes(h_dim, h_dim, activation))
        self.layer1a=nn.Sequential(*layer1a)
        self.layer1b=SinCos(h_dim, 2*h_dim, alpha, False)
        self.layer2=nn.Linear(2*h_dim, y_dim)
        self.layer2.weight.data*=gamma
        self.layer2.bias.data*=0

    def forward(self, x):
        x=preprocess(x, self.x_scale)
        x0=self.layer0(x)
        x1=self.layer1a(x0)
        x1=self.layer1b(x1)
        y=self.layer2(x1)
        return y
#%%
class BaseNet2(nn.Module):
    def __init__(self, x_dim, h_dim, n_layers, x_scale, alpha, gamma, y_dim, activation="softplus"):
        super().__init__()
        self.x_dim=x_dim
        self.h_dim=h_dim
        self.n_layers=n_layers
        self.x_scale=x_scale
        self.alpha=None
        self.gamma=gamma
        self.y_dim=y_dim
        self.activation=activation
        self.layer0=SinCos(x_dim, 2*h_dim, x_scale, True)
        layer1=[]
        layer1.append(LinearRes(2*h_dim, h_dim, activation))
        for n in range(1, n_layers):
            layer1.append(LinearRes(h_dim, h_dim, activation))
        self.layer1=nn.Sequential(*layer1)
        self.layer2=nn.Linear(h_dim, y_dim)
        self.layer2.weight.data*=gamma
        self.layer2.bias.data*=0

    def forward(self, x):
        x0=self.layer0(x)
        x1=self.layer1(x0)
        y=self.layer2(x1)
        return y
#%%
class BaseNet2a(nn.Module):
    def __init__(self, x_dim, h_dim, n_layers, x_scale, alpha, gamma, y_dim, activation="softplus"):
        super().__init__()
        self.x_dim=x_dim
        self.h_dim=h_dim
        self.n_layers=n_layers
        self.x_scale=x_scale
        self.alpha=None
        self.gamma=gamma
        self.y_dim=y_dim
        self.activation=activation
        self.layer0=SinCos(x_dim, 2*h_dim, x_scale, True)
        layer1=[]
        for n in range(0, n_layers):
            if n == 0:
                layer1.append(nn.Linear(2*h_dim, h_dim))
            else:
                layer1.append(nn.Linear(h_dim, h_dim))
            if activation == 'softplus':
                layer1.append(nn.Softplus())
            elif activation == 'relu':
                layer1.append(nn.ReLU())
            elif activation == "elu":
                layer1.append(nn.ELU())
            else:
                raise ValueError("unknown activation:"+activation)
        self.layer1=nn.Sequential(*layer1)
        self.layer2=nn.Linear(h_dim, y_dim)
        self.layer2.weight.data*=gamma
        self.layer2.bias.data*=0

    def forward(self, x):
        x0=self.layer0(x)
        x1=self.layer1(x0)
        y=self.layer2(x1)
        return y
#%%
class BaseNet2b(nn.Module):
    def __init__(self, x_dim, h_dim, n_layers, x_scale, alpha, gamma, y_dim, activation="softplus"):
        super().__init__()
        self.x_dim=x_dim
        self.h_dim=h_dim
        self.n_layers=n_layers
        self.x_scale=x_scale
        self.alpha=None
        self.gamma=gamma
        self.y_dim=y_dim
        self.activation=activation
        self.layer0=Sin(x_dim, h_dim, x_scale, True)
        layer1=[]
        for n in range(0, n_layers):
            layer1.append(nn.Linear(h_dim, h_dim))
            if activation == 'softplus':
                layer1.append(nn.Softplus())
            elif activation == 'relu':
                layer1.append(nn.ReLU())
            elif activation == "elu":
                layer1.append(nn.ELU())
            else:
                raise ValueError("unknown activation:"+activation)
        self.layer1=nn.Sequential(*layer1)
        self.layer2=nn.Linear(h_dim, y_dim)
        self.layer2.weight.data*=gamma
        self.layer2.bias.data*=0

    def forward(self, x):
        x0=self.layer0(x)
        x1=self.layer1(x0)
        y=self.layer2(x1)
        return y
#%%
class BaseNet3(nn.Module):
    def __init__(self, x_dim, h_dim, n_layers, x_scale, alpha, gamma, y_dim, activation="none"):
        super().__init__()
        self.x_dim=x_dim
        self.h_dim=h_dim
        self.n_layers=n_layers
        self.x_scale=x_scale
        self.alpha=alpha
        self.gamma=gamma
        self.y_dim=y_dim
        self.activation=activation
        self.layer0=SinCos(x_dim, 2*h_dim, x_scale, True)
        layer1=[]
        layer1.append(SinCosRes(2*h_dim, h_dim, alpha, False, activation))
        for n in range(1, n_layers):
            layer1.append(SinCosRes(h_dim, h_dim, alpha, False,  activation))
        self.layer1=nn.Sequential(*layer1)
        self.layer2=nn.Linear(h_dim, y_dim)
        self.layer2.weight.data*=gamma
        self.layer2.bias.data*=0

    def forward(self, x):
        x0=self.layer0(x)
        x1=self.layer1(x0)
        y=self.layer2(x1)
        return y
#%%
class BaseNet3b(nn.Module):
    def __init__(self, x_dim, h_dim, n_layers, x_scale, alpha, gamma, y_dim, activation="none"):
        super().__init__()
        self.x_dim=x_dim
        self.h_dim=h_dim
        self.n_layers=n_layers
        self.x_scale=x_scale
        self.alpha=alpha
        self.gamma=gamma
        self.y_dim=y_dim
        self.activation=activation
        self.layer0=Sin(x_dim, h_dim, x_scale, True)
        layer1=[]
        for n in range(0, n_layers):
            layer1.append(SinRes(h_dim, h_dim, alpha, False,  activation))
        self.layer1=nn.Sequential(*layer1)
        self.layer2=nn.Linear(h_dim, y_dim)
        self.layer2.weight.data*=gamma
        self.layer2.bias.data*=0

    def forward(self, x):
        x0=self.layer0(x)
        x1=self.layer1(x0)
        y=self.layer2(x1)
        return y
#%%
class BaseNet4(nn.Module):
    def __init__(self, x_dim, h_dim, n_layers, x_scale, alpha, gamma, y_dim):
        super().__init__()
        self.x_dim=x_dim
        self.h_dim=h_dim
        self.n_layers=n_layers
        self.x_scale=x_scale
        self.alpha=alpha
        self.gamma=gamma
        self.y_dim=y_dim
        self.layer0=nn.Sequential(nn.Linear(x_dim, h_dim),
                                  nn.Softplus(),
                                  nn.Linear(h_dim, h_dim),
                                  nn.Softplus())
        self.layer0a=SinCos(h_dim, h_dim, alpha, True)
        self.layer0b=nn.Linear(h_dim, h_dim)
        layer1a=[]
        layer1b=[]
        for n in range(0, n_layers):
            layer1a.append(SinCos(h_dim, h_dim, alpha, False))
            layer1b.append(nn.Linear(h_dim, h_dim))
        self.layer1a=nn.Sequential(*layer1a)
        self.layer1b=nn.Sequential(*layer1b)
        self.layer2=nn.Linear(h_dim, y_dim)
        self.layer2.weight.data*=gamma
        self.layer2.bias.data*=0

    def forward(self, x):
        x=preprocess(x, self.x_scale)
        x0=self.layer0(x)
        xx=self.layer0a(x0)
        x1=self.layer0b(xx)
        for fa, fb in zip(self.layer1a, self.layer1b):
            xx=fa(xx)
            x1=x1+fb(xx)
        y=self.layer2(x1)
        return y
#%%
class BaseNet4a(nn.Module):
    def __init__(self, x_dim, h_dim, n_layers, x_scale, alpha, gamma, y_dim):
        super().__init__()
        self.x_dim=x_dim
        self.h_dim=h_dim
        self.n_layers=n_layers
        self.x_scale=x_scale
        self.alpha=alpha
        self.gamma=gamma
        self.y_dim=y_dim
        self.layer0a=SinCos(x_dim, h_dim, x_scale, True)
        self.layer0b=nn.Linear(h_dim, h_dim)
        layer1a=[]
        layer1b=[]
        for n in range(0, n_layers):
            layer1a.append(SinCos(h_dim, h_dim, alpha, False))
            layer1b.append(nn.Linear(h_dim, h_dim))
        self.layer1a=nn.Sequential(*layer1a)
        self.layer1b=nn.Sequential(*layer1b)
        self.layer2=nn.Linear(h_dim, y_dim)
        self.layer2.weight.data*=gamma
        self.layer2.bias.data*=0

    def forward(self, x):
        x0=self.layer0a(x)
        x1=self.layer0b(x0)
        xx=x0
        for fa, fb in zip(self.layer1a, self.layer1b):
            xx=fa(xx)
            x1=x1+fb(xx)
        y=self.layer2(x1)
        return y
#%%
class BaseNet4b(nn.Module):
    def __init__(self, x_dim, h_dim, n_layers, x_scale, alpha, gamma, y_dim):
        super().__init__()
        self.x_dim=x_dim
        self.h_dim=h_dim
        self.n_layers=n_layers
        self.x_scale=x_scale
        self.alpha=alpha
        self.gamma=gamma
        self.y_dim=y_dim
        self.layer0=SinCos(x_dim, h_dim, x_scale, True)
        layer1=[]
        for n in range(0, n_layers):
            layer1.append(SinCos(h_dim, h_dim, alpha, False))
        self.layer1=nn.Sequential(*layer1)
        self.layer2=nn.Linear(h_dim, y_dim)
        self.layer2.weight.data*=gamma
        self.layer2.bias.data*=0

    def forward(self, x):
        x0=self.layer0(x)
        x1=self.layer1(x0)
        y=self.layer2(x1)
        return y
#%%
class BaseNet5a(nn.Module):
    def __init__(self, x_dim, h_dim, n_layers, x_scale, alpha, gamma, y_dim):
        super().__init__()
        self.x_dim=x_dim
        self.h_dim=h_dim
        self.n_layers=n_layers
        self.x_scale=x_scale
        self.alpha=alpha
        self.gamma=gamma
        self.y_dim=y_dim
        self.layer0a=Sin(x_dim, h_dim, x_scale, True)
        self.layer0b=nn.Linear(h_dim, h_dim)
        layer1a=[]
        layer1b=[]
        for n in range(0, n_layers):
            layer1a.append(Sin(h_dim, h_dim, alpha, False))
            layer1b.append(nn.Linear(h_dim, h_dim))
        self.layer1a=nn.Sequential(*layer1a)
        self.layer1b=nn.Sequential(*layer1b)
        self.layer2=nn.Linear(h_dim, y_dim)
        self.layer2.weight.data*=gamma
        self.layer2.bias.data*=0

    def forward(self, x):
        x0=self.layer0a(x)
        x1=self.layer0b(x0)
        xx=x0
        for fa, fb in zip(self.layer1a, self.layer1b):
            xx=fa(xx)
            x1=x1+fb(xx)
        y=self.layer2(x1)
        return y
#%%
class BaseNet5b(nn.Module):
    def __init__(self, x_dim, h_dim, n_layers, x_scale, alpha, gamma, y_dim):
        super().__init__()
        self.x_dim=x_dim
        self.h_dim=h_dim
        self.n_layers=n_layers
        self.x_scale=x_scale
        self.alpha=alpha
        self.gamma=gamma
        self.y_dim=y_dim
        self.layer0=Sin(x_dim, h_dim, x_scale, True)
        layer1=[]
        for n in range(0, n_layers):
            layer1.append(Sin(h_dim, h_dim, alpha, False))
        self.layer1=nn.Sequential(*layer1)
        self.layer2=nn.Linear(h_dim, y_dim)
        self.layer2.weight.data*=gamma
        self.layer2.bias.data*=0

    def forward(self, x):
        x0=self.layer0(x)
        x1=self.layer1(x0)
        y=self.layer2(x1)
        return y
#%%
class BaseNet6c(nn.Module):
    def __init__(self, x_dim, h_dim, n_layers, x_scale, alpha, gamma, y_dim):
        super().__init__()
        self.x_dim=x_dim
        self.h_dim=h_dim
        self.n_layers=n_layers
        self.x_scale=x_scale
        self.alpha=alpha
        self.gamma=gamma
        self.y_dim=y_dim
        self.layer0=Sin(x_dim, h_dim, x_scale, True)
        layer1aa=[]
        layer1ab=[]
        for n in range(0, n_layers):
            layer1aa.append(Sin(h_dim, h_dim, alpha, False))
            layer1ab.append(nn.Linear(h_dim, h_dim))
        layer1b=[]
        for n in range(0, n_layers-1):
            layer1b.append(LinearRes(h_dim, h_dim, "softplus"))
        layer1b.append(nn.Linear(h_dim, h_dim))
        self.layer1aa=nn.Sequential(*layer1aa)
        self.layer1ab=nn.Sequential(*layer1ab)
        self.layer1b=nn.Sequential(*layer1b)
        self.layer2=nn.Linear(h_dim, y_dim)
        self.layer2.weight.data*=gamma
        self.layer2.bias.data*=0

    def forward(self, x):
        x0=self.layer0(x)
        xx=x0
        x1a=0
        for fa, fb in zip(self.layer1aa, self.layer1ab):
            xx=fa(xx)
            x1a=x1a+fb(xx)
        x1b=self.layer1b(x0)
        x1=x1a*x1b
        y=self.layer2(x1)
        return y
#%%
class BaseNet6a(nn.Module):
    def __init__(self, x_dim, h_dim, n_layers, x_scale, alpha, gamma, y_dim):
        super().__init__()
        self.x_dim=x_dim
        self.h_dim=h_dim
        self.n_layers=n_layers
        self.x_scale=x_scale
        self.alpha=alpha
        self.gamma=gamma
        self.y_dim=y_dim
        self.layer0=SinCos(x_dim, h_dim, x_scale, True)
        layer1a=[]
        for n in range(0, n_layers):
            layer1a.append(SinCos(h_dim, h_dim, alpha, False))
        layer1b=[]
        for n in range(0, n_layers-1):
            layer1b.append(LinearRes(h_dim, h_dim, "softplus"))
        layer1b.append(nn.Linear(h_dim, h_dim*h_dim))
        self.layer1a=nn.Sequential(*layer1a)
        self.layer1b=nn.Sequential(*layer1b)
        self.layer2=nn.Linear(h_dim, y_dim)
        self.layer2.weight.data*=gamma
        self.layer2.bias.data*=0

    def forward(self, x):
        x0=self.layer0(x)
        x1a=self.layer1a(x0)
        x1b=self.layer1b(x0)
        x1a=x1a.view(-1, 1, self.h_dim)
        x1b=x1b.view(-1, self.h_dim, self.h_dim)
        x1=torch.bmm(x1a, x1b)
        x1=x1.view(-1,self.h_dim)
        y=self.layer2(x1)
        return y
#%%
class BaseNet6b(nn.Module):
    def __init__(self, x_dim, h_dim, n_layers, x_scale, alpha, gamma, y_dim):
        super().__init__()
        self.x_dim=x_dim
        self.h_dim=h_dim
        self.n_layers=n_layers
        self.x_scale=x_scale
        self.alpha=alpha
        self.gamma=gamma
        self.y_dim=y_dim
        self.layer0=Sin(x_dim, h_dim, x_scale, True)
        layer1a=[]
        for n in range(0, n_layers):
            layer1a.append(Sin(h_dim, h_dim, alpha, False))
        layer1b=[]
        for n in range(0, n_layers-1):
            layer1b.append(LinearRes(h_dim, h_dim, "softplus"))
        layer1b.append(nn.Linear(h_dim, h_dim*h_dim))
        self.layer1a=nn.Sequential(*layer1a)
        self.layer1b=nn.Sequential(*layer1b)
        self.layer2=nn.Linear(h_dim, y_dim)
        self.layer2.weight.data*=gamma
        self.layer2.bias.data*=0

    def forward(self, x):
        x0=self.layer0(x)
        x1a=self.layer1a(x0)
        x1b=self.layer1b(x0)
        x1a=x1a.view(-1, 1, self.h_dim)
        x1b=x1b.view(-1, self.h_dim, self.h_dim)
        x1=torch.bmm(x1a, x1b)
        x1=x1.view(-1,self.h_dim)
        y=self.layer2(x1)
        return y
