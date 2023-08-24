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
class SinRes(nn.Module):
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
class Net0(nn.Module):
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
        if activation == 'softplus':
            self.layer0=nn.Sequential(nn.Linear(x_dim, h_dim),
                                      nn.Softplus())
        elif activation == 'relu':
            self.layer0=nn.Sequential(nn.Linear(x_dim, h_dim),
                                      nn.ReLU())
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
class Net1(nn.Module):
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
        layer1=[]
        for n in range(0, n_layers):
            layer1.append(LinearRes(h_dim, h_dim, activation))
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
class Net2(nn.Module):
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
class Net2a(nn.Module):
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
class Net2b(nn.Module):
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
class Net3(nn.Module):
    def __init__(self, x_dim, h_dim, n_layers, x_scale, alpha, gamma, y_dim, freeze_layer0=False):
        super().__init__()
        self.x_dim=x_dim
        self.h_dim=h_dim
        self.n_layers=n_layers
        self.x_scale=x_scale
        self.alpha=alpha
        self.gamma=gamma
        self.y_dim=y_dim
        self.freeze_layer0=freeze_layer0
        self.layer0=Sin(x_dim, h_dim, x_scale, True)
        if freeze_layer0 == True:
            for p in self.layer0.parameters():
                p.requires_grad=False
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
class MLP1(nn.Module):
    def __init__(self, x_dim, h_dim, n_layers, y_dim, x_scale=1, layer_norm=False, activation='softplus'):
        super().__init__()
        self.x_dim=x_dim
        self.h_dim=h_dim
        self.y_dim=y_dim
        self.x_scale=x_scale
        self.layer_norm=layer_norm
        self.activation=activation
        if self.activation == 'softplus':
            self.layer0=nn.Sequential(nn.Linear(x_dim, h_dim),
                                      nn.Softplus())
        elif self.activation == 'relu':
            self.layer0=nn.Sequential(nn.Linear(x_dim, h_dim),
                                      nn.ReLU())
        else:
            raise ValueError('error')
        layer1=[]
        for n in range(0, n_layers):
            layer1.append(nn.Linear(h_dim, h_dim))
            if self.activation == 'softplus':
                layer1.append(nn.Softplus())
                if layer_norm == True:
                    layer1.append(nn.GroupNorm(1, h_dim))
            elif self.activation == 'relu':
                layer1.append(nn.ReLU())
                if layer_norm == True:
                    layer1.append(nn.GroupNorm(1, h_dim))
            elif self.activation == 'softplus_sigmoid':
                if n < n_layers-1:
                    layer1.append(nn.Softplus())
                    if layer_norm == True:
                        layer1.append(nn.GroupNorm(1, h_dim))
                else:
                    layer1.append(nn.GroupNorm(1,h_dim))
                    layer1.append(nn.Sigmoid())
            else:
                raise ValueError('error')
        self.layer1=nn.Sequential(*layer1)
        self.layer2=nn.Linear(h_dim, y_dim)
    def forward(self, x):
        #x.shape (N, 3)
        if self.x_scale != 1:
            x=x*self.x_scale
        x0=self.layer0(x)
        x1=self.layer1(x0)
        y=self.layer2(x1)#(N, y_dim)
        return y

