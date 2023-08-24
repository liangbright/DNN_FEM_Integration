import torch
import torch.nn as nn
import torch.nn.functional as nnF
from NNFEA_net_base import LinearRes, Sin, preprocess
from NNFEA_net_base import BaseNet0, BaseNet1, BaseNet1a, BaseNet2, BaseNet2a, BaseNet2b, BaseNet3, BaseNet3b
from NNFEA_net_base import BaseNet4, BaseNet4a, BaseNet4b, BaseNet5a, BaseNet5b
from NNFEA_net_base import BaseNet6a, BaseNet6b
#%%
def predict(x, c, netA, netB, h_dim, y_dim):
    #x.shape (N, x_dim)
    #c.shape (N, c_dim) or (1, c_dim)
    N=x.shape[0]
    y1=netA(x)
    y2=netB(c)
    y1=y1.view(N, y_dim, h_dim)
    y2=y2.view(y2.shape[0], h_dim, 1).expand(N, h_dim, 1)
    y=torch.bmm(y1,y2)
    y=y.view(N, y_dim)
    return y
#%%
def predict2(x, c, netA, netB, h_dim, y_dim):
    #x.shape (N, x_dim)
    #c.shape (N, c_dim)
    N=x.shape[0]
    xc=torch.cat([x,c],dim=1)
    y1=netA(xc)
    y2=netB(c)
    y1=y1.view(N, y_dim, h_dim)
    y2=y2.view(N, h_dim, 1)
    y=torch.bmm(y1,y2)
    y=y.view(N, y_dim)
    return y
#%%
def predict2a(x, c, netA, netB, h_dim, y_dim):
    #x.shape (N, x_dim)
    #c.shape (N, c_dim)
    N=x.shape[0]
    xc=torch.cat([x,c],dim=1)
    y1=netA(xc)
    y2=netB(xc)
    y1=y1.view(N, y_dim, h_dim)
    y2=y2.view(N, h_dim, 1)
    y=torch.bmm(y1,y2)
    y=y.view(N, y_dim)
    return y
#%%
class Net(nn.Module):
    def __init__(self, BaseNet, x_dim, c_dim, h_dim, n_layers, x_scale, alpha, gamma, y_dim):
        super().__init__()
        self.BaseNet=BaseNet
        self.x_dim=x_dim
        self.c_dim=c_dim
        self.h_dim=h_dim
        self.n_layers=n_layers
        self.x_scale=x_scale
        self.alpha=alpha
        self.gamma=gamma
        self.y_dim=y_dim
        self.netA=eval(BaseNet+"("
                       +'x_dim='+str(x_dim)
                       +',h_dim='+str(h_dim)
                       +',n_layers='+str(n_layers)
                       +',x_scale='+str(x_scale)
                       +',alpha='+str(alpha)
                       +',gamma='+str(gamma)
                       +',y_dim='+str(y_dim*h_dim)
                       +")"
                       )
        self.netB=nn.Sequential(nn.Linear(c_dim, h_dim),
                                nn.Softplus(),
                                nn.Linear(h_dim, h_dim),
                                nn.Softplus(),
                                nn.Linear(h_dim, h_dim))

    def forward(self, x, c):
        y=predict(x, c, self.netA, self.netB, self.h_dim, self.y_dim)
        return y
#%%
class Net1a(nn.Module):
    def __init__(self, BaseNet, x_dim, c_dim, h_dim, n_layers, x_scale, alpha, gamma, y_dim):
        super().__init__()
        self.BaseNet=BaseNet
        self.x_dim=x_dim
        self.c_dim=c_dim
        self.h_dim=h_dim
        self.n_layers=n_layers
        self.x_scale=x_scale
        self.alpha=alpha
        self.gamma=gamma
        self.y_dim=y_dim
        self.netA=eval(BaseNet+"("
                       +'x_dim='+str(x_dim)
                       +',h_dim='+str(h_dim)
                       +',n_layers='+str(n_layers)
                       +',x_scale='+str(x_scale)
                       +',alpha='+str(alpha)
                       +',gamma='+str(gamma)
                       +',y_dim='+str(y_dim*h_dim)
                       +")"
                       )
        self.netB=nn.Sequential(nn.Linear(c_dim, h_dim),
                                nn.Softplus(),
                                nn.Linear(h_dim, h_dim),
                                nn.Softplus(),
                                nn.Linear(h_dim, h_dim),
                                nn.Tanh())

    def forward(self, x, c, return_y_netB=False):
        if return_y_netB == False:
            y=predict(x, c, self.netA, self.netB, self.h_dim, self.y_dim)
        else:
            y=self.netB(c)
        return y
#%%
class Net1b(nn.Module):
    def __init__(self, BaseNet, x_dim, c_dim, h_dim, n_layers, x_scale, alpha, gamma, y_dim):
        super().__init__()
        self.BaseNet=BaseNet
        self.x_dim=x_dim
        self.c_dim=c_dim
        self.h_dim=h_dim
        self.n_layers=n_layers
        self.x_scale=x_scale
        self.alpha=alpha
        self.gamma=gamma
        self.y_dim=y_dim
        self.netA=eval(BaseNet+"("
                       +'x_dim='+str(x_dim)
                       +',h_dim='+str(h_dim)
                       +',n_layers='+str(n_layers)
                       +',x_scale='+str(x_scale)
                       +',alpha='+str(alpha)
                       +',gamma='+str(gamma)
                       +',y_dim='+str(y_dim*h_dim)
                       +")"
                       )
        self.netB=nn.Sequential(nn.Linear(c_dim, h_dim),
                                nn.Softplus(),
                                nn.Linear(h_dim, h_dim),
                                nn.Softplus(),
                                nn.Linear(h_dim, h_dim),
                                nn.Sigmoid())

    def forward(self, x, c, return_y_netB=False):
        if return_y_netB == False:
            y=predict(x, c, self.netA, self.netB, self.h_dim, self.y_dim)
        else:
            y=self.netB(c)
        return y
#%%
class Net1c(nn.Module):
    def __init__(self, BaseNet, x_dim, c_dim, h_dim, n_layers, x_scale, alpha, gamma, y_dim):
        super().__init__()
        self.BaseNet=BaseNet
        self.x_dim=x_dim
        self.c_dim=c_dim
        self.h_dim=h_dim
        self.n_layers=n_layers
        self.x_scale=x_scale
        self.alpha=alpha
        self.gamma=gamma
        self.y_dim=y_dim
        self.netA=eval(BaseNet+"("
                       +'x_dim='+str(x_dim)
                       +',h_dim='+str(h_dim)
                       +',n_layers='+str(n_layers)
                       +',x_scale='+str(x_scale)
                       +',alpha='+str(alpha)
                       +',gamma='+str(gamma)
                       +',y_dim='+str(y_dim*h_dim)
                       +")"
                       )
        self.netB=nn.Sequential(nn.Linear(c_dim, h_dim),
                                nn.Softplus(),
                                nn.Linear(h_dim, h_dim),
                                nn.Softplus(),
                                nn.Linear(h_dim, h_dim),
                                nn.Softplus())

    def forward(self, x, c):
        y=predict(x, c, self.netA, self.netB, self.h_dim, self.y_dim)
        return y
#%%
class Net1d(nn.Module):
    def __init__(self, BaseNet, x_dim, c_dim, h_dim, n_layers, x_scale, alpha, gamma, y_dim):
        super().__init__()
        self.BaseNet=BaseNet
        self.x_dim=x_dim
        self.c_dim=c_dim
        self.h_dim=h_dim
        self.n_layers=n_layers
        self.x_scale=x_scale
        self.alpha=alpha
        self.gamma=gamma
        self.y_dim=y_dim
        self.netA=eval(BaseNet+"("
                       +'x_dim='+str(x_dim)
                       +',h_dim='+str(h_dim)
                       +',n_layers='+str(n_layers)
                       +',x_scale='+str(x_scale)
                       +',alpha='+str(alpha)
                       +',gamma='+str(gamma)
                       +',y_dim='+str(y_dim*h_dim)
                       +")"
                       )
        self.netB=nn.Sequential(nn.Linear(c_dim, h_dim),
                                nn.Softplus(),
                                nn.Linear(h_dim, h_dim),
                                nn.Softplus(),
                                nn.Linear(h_dim, h_dim),
                                nn.Softmax(dim=-1))

    def forward(self, x, c):
        y=predict(x, c, self.netA, self.netB, self.h_dim, self.y_dim)
        return y
#%%
class Net1(nn.Module):
    def __init__(self, BaseNet, x_dim, c_dim, h_dim, n_layers, x_scale, alpha, gamma, y_dim, activation='sigmoid'):
        super().__init__()
        self.BaseNet=BaseNet
        self.x_dim=x_dim
        self.c_dim=c_dim
        self.h_dim=h_dim
        self.n_layers=n_layers
        self.x_scale=x_scale
        self.alpha=alpha
        self.gamma=gamma
        self.y_dim=y_dim
        self.netA=eval(BaseNet+"("
                       +'x_dim='+str(x_dim)
                       +',h_dim='+str(h_dim)
                       +',n_layers='+str(n_layers)
                       +',x_scale='+str(x_scale)
                       +',alpha='+str(alpha)
                       +',gamma='+str(gamma)
                       +',y_dim='+str(y_dim*h_dim)
                       +")"
                       )
        self.activation=activation
        if activation =='none':
            self.netB=nn.Sequential(nn.Linear(c_dim, h_dim),
                                    nn.Softplus(),
                                    nn.Linear(h_dim, h_dim),
                                    nn.Softplus(),
                                    nn.Linear(h_dim, h_dim))
        elif activation == 'sigmoid':
            self.netB=nn.Sequential(nn.Linear(c_dim, h_dim),
                                    nn.Softplus(),
                                    nn.Linear(h_dim, h_dim),
                                    nn.Softplus(),
                                    nn.Linear(h_dim, h_dim),
                                    nn.Sigmoid())
        elif activation == 'softplus':
            self.netB=nn.Sequential(nn.Linear(c_dim, h_dim),
                                    nn.Softplus(),
                                    nn.Linear(h_dim, h_dim),
                                    nn.Softplus(),
                                    nn.Linear(h_dim, h_dim),
                                    nn.Softplus())
        elif activation == 'softmax':
            self.netB=nn.Sequential(nn.Linear(c_dim, h_dim),
                                    nn.Softplus(),
                                    nn.Linear(h_dim, h_dim),
                                    nn.Softplus(),
                                    nn.Linear(h_dim, h_dim),
                                    nn.Softmax(dim=-1))
        elif activation == 'tanh':
            self.netB=nn.Sequential(nn.Linear(c_dim, h_dim),
                                    nn.Softplus(),
                                    nn.Linear(h_dim, h_dim),
                                    nn.Softplus(),
                                    nn.Linear(h_dim, h_dim),
                                    nn.Tanh())
        else:
            raise ValueError('unknown activation:'+str(activation))

    def forward(self, x, c):
        y=predict(x, c, self.netA, self.netB, self.h_dim, self.y_dim)
        return y
#%%
class Net2(nn.Module):
    def __init__(self, BaseNet, x_dim, c_dim, h_dim, n_layers, x_scale, alpha, gamma, y_dim, activation='sigmoid'):
        super().__init__()
        self.BaseNet=BaseNet
        self.x_dim=x_dim
        self.c_dim=c_dim
        self.h_dim=h_dim
        self.n_layers=n_layers
        self.x_scale=x_scale
        self.alpha=alpha
        self.gamma=gamma
        self.y_dim=y_dim
        self.netA=eval(BaseNet+"("
                       +'x_dim='+str(x_dim+c_dim)
                       +',h_dim='+str(h_dim)
                       +',n_layers='+str(n_layers)
                       +',x_scale=1'
                       +',alpha='+str(alpha)
                       +',gamma='+str(gamma)
                       +',y_dim='+str(y_dim*h_dim)
                       +")"
                       )
        self.activation=activation
        if activation =='none':
            self.netB=nn.Sequential(nn.Linear(c_dim, h_dim),
                                    nn.Softplus(),
                                    nn.Linear(h_dim, h_dim),
                                    nn.Softplus(),
                                    nn.Linear(h_dim, h_dim))
        elif activation == 'sigmoid':
            self.netB=nn.Sequential(nn.Linear(c_dim, h_dim),
                                    nn.Softplus(),
                                    nn.Linear(h_dim, h_dim),
                                    nn.Softplus(),
                                    nn.Linear(h_dim, h_dim),
                                    nn.Sigmoid())
        elif activation == 'softplus':
            self.netB=nn.Sequential(nn.Linear(c_dim, h_dim),
                                    nn.Softplus(),
                                    nn.Linear(h_dim, h_dim),
                                    nn.Softplus(),
                                    nn.Linear(h_dim, h_dim),
                                    nn.Softplus())
        elif activation == 'softmax':
            self.netB=nn.Sequential(nn.Linear(c_dim, h_dim),
                                    nn.Softplus(),
                                    nn.Linear(h_dim, h_dim),
                                    nn.Softplus(),
                                    nn.Linear(h_dim, h_dim),
                                    nn.Softmax(dim=-1))
        else:
            raise ValueError('unknown activation:'+str(activation))

    def forward(self, x, c):
        x=x*self.x_scale
        y=predict2(x, c, self.netA, self.netB, self.h_dim, self.y_dim)
        return y
#%% ok
class Net2a(nn.Module):
    def __init__(self, BaseNet, x_dim, c_dim, h_dim, n_layers, x_scale, alpha, gamma, y_dim, activation='sigmoid'):
        super().__init__()
        self.BaseNet=BaseNet
        self.x_dim=x_dim
        self.c_dim=c_dim
        self.h_dim=h_dim
        if isinstance(n_layers, float) or isinstance(n_layers, int):
            n_layers=(n_layers, n_layers)
        self.n_layers=n_layers
        self.x_scale=x_scale
        self.alpha=alpha
        self.gamma=gamma
        self.y_dim=y_dim
        self.netA=eval(BaseNet+"("
                       +'x_dim='+str(x_dim+c_dim)
                       +',h_dim='+str(h_dim)
                       +',n_layers='+str(n_layers[0])
                       +',x_scale='+str(x_scale)
                       +',alpha='+str(alpha)
                       +',gamma='+str(gamma)
                       +',y_dim='+str(y_dim*h_dim)
                       +")"
                       )
        self.netB=eval(BaseNet+"("
                       +'x_dim='+str(x_dim+c_dim)
                       +',h_dim='+str(h_dim)
                       +',n_layers='+str(n_layers[1])
                       +',x_scale='+str(x_scale)
                       +',alpha='+str(alpha)
                       +',gamma='+str(gamma)
                       +',y_dim='+str(h_dim)
                       +")"
                       )
        self.activation=activation
        if activation =='none':
            pass
        elif activation == 'sigmoid':
            self.netB=nn.Sequential(self.netB, nn.Sigmoid())
        elif activation == 'softmax':
            self.netB=nn.Sequential(self.netB, nn.Softmax(dim=-1))
        else:
            raise ValueError('unknown activation:'+str(activation))

    def forward(self, x, c):
        y=predict2a(x, c, self.netA, self.netB, self.h_dim, self.y_dim)
        return y
#%% not good
class Net2b(nn.Module):
    def __init__(self, BaseNet, x_dim, c_dim, h_dim, n_layers, x_scale, alpha, gamma, y_dim, activation='sigmoid'):
        super().__init__()
        self.BaseNet=BaseNet
        self.x_dim=x_dim
        self.c_dim=c_dim
        self.h_dim=h_dim
        if isinstance(n_layers, float) or isinstance(n_layers, int):
            n_layers=(n_layers, n_layers)
        self.n_layers=n_layers
        self.x_scale=x_scale
        self.alpha=alpha
        self.gamma=gamma
        self.y_dim=y_dim
        self.netA=eval(BaseNet+"("
                       +'x_dim='+str(x_dim+c_dim)
                       +',h_dim='+str(h_dim)
                       +',n_layers='+str(n_layers[0])
                       +',x_scale='+str(x_scale)
                       +',alpha='+str(alpha)
                       +',gamma='+str(gamma)
                       +',y_dim='+str(y_dim*h_dim)
                       +")"
                       )
        self.netB=eval(BaseNet+"("
                       +'x_dim='+str(c_dim)
                       +',h_dim='+str(h_dim)
                       +',n_layers='+str(n_layers[1])
                       +',x_scale='+str(x_scale)
                       +',alpha='+str(alpha)
                       +',gamma='+str(gamma)
                       +',y_dim='+str(h_dim)
                       +")"
                       )
        self.activation=activation
        if activation =='none':
            pass
        elif activation == 'sigmoid':
            self.netB=nn.Sequential(self.netB, nn.Sigmoid())
        elif activation == 'softmax':
            self.netB=nn.Sequential(self.netB, nn.Softmax(dim=-1))
        else:
            raise ValueError('unknown activation:'+str(activation))

    def forward(self, x, c):
        y=predict2(x, c, self.netA, self.netB, self.h_dim, self.y_dim)
        return y
#%% not good, not converge
class Net2c(nn.Module):
    def __init__(self, BaseNet, x_dim, c_dim, h_dim, n_layers, x_scale, alpha, gamma, y_dim, activation='sigmoid'):
        super().__init__()
        self.BaseNet=BaseNet
        self.x_dim=x_dim
        self.c_dim=c_dim
        self.h_dim=h_dim
        if isinstance(n_layers, float) or isinstance(n_layers, int):
            n_layers=(n_layers, n_layers)
        self.n_layers=n_layers
        self.x_scale=x_scale
        self.alpha=alpha
        self.gamma=gamma
        self.y_dim=y_dim
        self.netA=eval(BaseNet+"("
                       +'x_dim='+str(x_dim)
                       +',h_dim='+str(h_dim)
                       +',n_layers='+str(n_layers[0])
                       +',x_scale='+str(x_scale)
                       +',alpha='+str(alpha)
                       +',gamma='+str(gamma)
                       +',y_dim='+str(y_dim*h_dim)
                       +")"
                       )
        self.netB=eval(BaseNet+"("
                       +'x_dim='+str(c_dim)
                       +',h_dim='+str(h_dim)
                       +',n_layers='+str(n_layers[1])
                       +',x_scale='+str(x_scale)
                       +',alpha='+str(alpha)
                       +',gamma='+str(gamma)
                       +',y_dim='+str(h_dim)
                       +")"
                       )
        self.activation=activation
        if activation =='none':
            pass
        elif activation == 'sigmoid':
            self.netB=nn.Sequential(self.netB, nn.Sigmoid())
        elif activation == 'softmax':
            self.netB=nn.Sequential(self.netB, nn.Softmax(dim=-1))
        else:
            raise ValueError('unknown activation:'+str(activation))

    def forward(self, x, c):
        y=predict(x, c, self.netA, self.netB, self.h_dim, self.y_dim)
        return y
#%%
from Attention import ScaledDotProductAttention
class Net3(nn.Module):
    def __init__(self, BaseNet, x_dim, c_dim, h_dim, n_layers, x_scale, alpha, gamma, y_dim, activation="sigmoid"):
        super().__init__()
        self.BaseNet=BaseNet
        self.x_dim=x_dim
        self.c_dim=c_dim
        if isinstance(h_dim, float) or isinstance(h_dim, int):
            k_dim=h_dim
        else:
            k_dim=h_dim[1]
            h_dim=h_dim[0]
        self.k_dim=k_dim
        self.h_dim=h_dim
        if isinstance(n_layers, float) or isinstance(n_layers, int):
            n_layers=(n_layers, n_layers, n_layers)
        self.n_layers=n_layers
        self.x_scale=x_scale
        self.alpha=alpha
        self.gamma=gamma
        self.y_dim=y_dim
        self.net_q=eval(BaseNet+"("
                       +'x_dim='+str(x_dim+c_dim)
                       +',h_dim='+str(h_dim)
                       +',n_layers='+str(n_layers[0])
                       +',x_scale='+str(x_scale)
                       +',alpha='+str(alpha)
                       +',gamma='+str(gamma)
                       +',y_dim='+str(k_dim)
                       +")"
                       )
        self.net_k=eval(BaseNet+"("
                       +'x_dim='+str(x_dim+c_dim)
                       +',h_dim='+str(h_dim)
                       +',n_layers='+str(n_layers[1])
                       +',x_scale='+str(x_scale)
                       +',alpha='+str(alpha)
                       +',gamma='+str(gamma)
                       +',y_dim='+str(h_dim*k_dim)
                       +")"
                       )
        self.net_v=eval(BaseNet+"("
                       +'x_dim='+str(x_dim+c_dim)
                       +',h_dim='+str(h_dim)
                       +',n_layers='+str(n_layers[2])
                       +',x_scale='+str(x_scale)
                       +',alpha='+str(alpha)
                       +',gamma='+str(gamma)
                       +',y_dim='+str(h_dim*y_dim)
                       +")"
                       )
        self.attention=ScaledDotProductAttention(activation)

    def forward(self, x, c):
        xc=torch.cat([x,c],dim=1)
        q=self.net_q(xc)
        k=self.net_k(xc)
        v=self.net_v(xc)
        #---------------
        N=q.shape[0]
        q=q.view(N, 1, self.k_dim)
        k=k.view(N, self.h_dim, self.k_dim)
        v=v.view(N, self.h_dim, self.y_dim)
        y=self.attention(q, k, v)
        y=y.view(N, self.y_dim)
        return y
#%%
class Net4(nn.Module):
    def __init__(self, BaseNet, x_dim, c_dim, h_dim, n_layers, x_scale, alpha, gamma, y_dim, activation='sigmoid'):
        super().__init__()
        self.BaseNet=BaseNet
        self.x_dim=x_dim
        self.c_dim=c_dim
        self.h_dim=h_dim
        self.n_layers=n_layers
        self.x_scale=x_scale
        self.alpha=alpha
        self.gamma=gamma
        self.y_dim=y_dim
        self.netA=eval(BaseNet+"("
                       +'x_dim='+str(2*x_dim+c_dim)
                       +',h_dim='+str(h_dim)
                       +',n_layers='+str(n_layers)
                       +',x_scale=1'
                       +',alpha='+str(alpha)
                       +',gamma='+str(gamma)
                       +',y_dim='+str(y_dim*h_dim)
                       +")"
                       )
        self.activation=activation
        if activation =='none':
            self.netB=nn.Sequential(nn.Linear(c_dim, h_dim),
                                    nn.Softplus(),
                                    nn.Linear(h_dim, h_dim),
                                    nn.Softplus(),
                                    nn.Linear(h_dim, h_dim))
        elif activation == 'sigmoid':
            self.netB=nn.Sequential(nn.Linear(c_dim, h_dim),
                                    nn.Softplus(),
                                    nn.Linear(h_dim, h_dim),
                                    nn.Softplus(),
                                    nn.Linear(h_dim, h_dim),
                                    nn.Sigmoid())
        elif activation == 'softmax':
            self.netB=nn.Sequential(nn.Linear(c_dim, h_dim),
                                    nn.Softplus(),
                                    nn.Linear(h_dim, h_dim),
                                    nn.Softplus(),
                                    nn.Linear(h_dim, h_dim),
                                    nn.Softmax(dim=-1))
        else:
            raise ValueError('unknown activation:'+str(activation))

    def forward(self, x, x_ref, c):
        N=x.shape[0]
        #x=x*self.x_scale
        #x_ref=x_ref*self.x_scale
        xc=torch.cat([x, x_ref, c*self.x_scale],dim=1)
        y1=self.netA(xc)
        y2=self.netB(c)
        y1=y1.view(N, self.y_dim, self.h_dim)
        y2=y2.view(N, self.h_dim, 1)
        y=torch.bmm(y1,y2)
        y=y.view(N, self.y_dim)
        return y
#%%
class Net5(nn.Module):
    def __init__(self, BaseNet, x_dim, c_dim, h_dim, n_layers, x_scale, alpha, gamma, y_dim, activation='sigmoid'):
        super().__init__()
        self.BaseNet=BaseNet
        self.x_dim=x_dim
        self.c_dim=c_dim
        self.h_dim=h_dim
        self.n_layers=n_layers
        self.x_scale=x_scale
        self.alpha=alpha
        self.gamma=gamma
        self.y_dim=y_dim
        self.netA=eval(BaseNet+"("
                       +'x_dim='+str(2*x_dim)
                       +',h_dim='+str(h_dim)
                       +',n_layers='+str(n_layers)
                       +',x_scale='+str(x_scale)
                       +',alpha='+str(alpha)
                       +',gamma='+str(gamma)
                       +',y_dim='+str(y_dim*h_dim)
                       +")"
                       )
        self.activation=activation
        if activation =='none':
            self.netB=nn.Sequential(nn.Linear(c_dim, h_dim),
                                    nn.Softplus(),
                                    nn.Linear(h_dim, h_dim),
                                    nn.Softplus(),
                                    nn.Linear(h_dim, h_dim))
        elif activation == 'sigmoid':
            self.netB=nn.Sequential(nn.Linear(c_dim, h_dim),
                                    nn.Softplus(),
                                    nn.Linear(h_dim, h_dim),
                                    nn.Softplus(),
                                    nn.Linear(h_dim, h_dim),
                                    nn.Sigmoid())
        elif activation == 'softmax':
            self.netB=nn.Sequential(nn.Linear(c_dim, h_dim),
                                    nn.Softplus(),
                                    nn.Linear(h_dim, h_dim),
                                    nn.Softplus(),
                                    nn.Linear(h_dim, h_dim),
                                    nn.Softmax(dim=-1))
        else:
            raise ValueError('unknown activation:'+str(activation))

    def forward(self, x, x_ref, c):
        N=x.shape[0]
        xx=torch.cat([x, x_ref],dim=1)
        y1=self.netA(xx)
        y2=self.netB(c)
        y1=y1.view(N, self.y_dim, self.h_dim)
        y2=y2.view(N, self.h_dim, 1)
        y=torch.bmm(y1,y2)
        y=y.view(N, self.y_dim)
        return y
#%%
class EnsembleNet(nn.Module):
    def __init__(self, n_models, BaseNet, x_dim, c_dim, h_dim, n_layers, x_scale, alpha, gamma, y_dim, mode):
        super().__init__()
        self.n_models=n_models
        self.BaseNet=BaseNet
        self.x_dim=x_dim
        self.c_dim=c_dim
        self.h_dim=h_dim
        self.n_layers=n_layers
        self.x_scale=x_scale
        self.alpha=alpha
        self.gamma=gamma
        self.y_dim=y_dim
        self.mode=mode
        self.model_list=nn.ModuleList()
        self.model_flag=[]
        for n in range(0, n_models):
            model=eval("Net("+BaseNet
                       +',x_dim='+str(x_dim)
                       +',h_dim='+str(h_dim)
                       +',n_layers='+str(n_layers)
                       +',x_scale='+str(x_scale)
                       +',alpha='+str(alpha)
                       +',gamma='+str(gamma)
                       +',y_dim='+str(y_dim*h_dim)
                       +")"
                       )
            self.model_list.append(model)
        self.active_model=0

    def add_model(self, n=1):
        if self.active_model < len(self.model_list):
            self.active_model+=n

    def freeze_model(self, id):
        for param in self.model_list[id].parameters():
            param.requires_grad = False
            param.grad=None

    def unfreeze_model(self, id):
        for param in self.model_list[id].parameters():
            param.requires_grad = True

    def forward(self, x, c):
        if self.training == True:
            y_list=[]
            for n in range(0, self.active_model):
                model= self.model_list[n]
                y_list.append(model(x,c))
            return y_list
        else:
            y=0
            for n in range(0, self.active_model):
                model= self.model_list[n]
                y=y+model(x,c)
            if self.mode=='bagging':
                y=y/self.active_model
            return y
#%%
class MLP(nn.Module):
    def __init__(self, c_dim, h_dim, n_layers, y_dim):
        super().__init__()
        self.c_dim=c_dim
        self.h_dim=h_dim
        self.y_dim=y_dim
        self.layer0=LinearRes(c_dim, h_dim, "softplus")
        layer1=[]
        for n in range(0, n_layers):
            layer1.append(LinearRes(h_dim, h_dim, "softplus"))
        self.layer1=nn.Sequential(*layer1)
        self.layer2=nn.Linear(h_dim, y_dim)
    def forward(self, x, c):
        #x.shape (N, 3)
        #c.shape (N, c_dim)
        #y_dim should be N*3
        c=c[0:1]
        c0=self.layer0(c)
        c1=self.layer1(c0)
        c2=self.layer2(c1)
        #print(x.shape, c2.shape)
        y=c2.view(x.shape[0], -1)
        return y
#%%
class MLP1(nn.Module):
    def __init__(self, c_dim, h_dim, n_layers, y_dim, activation='softplus'):
        super().__init__()
        self.c_dim=c_dim
        self.h_dim=h_dim
        self.y_dim=y_dim
        self.activation=activation
        self.layer0=nn.Linear(c_dim, h_dim)
        layer1=[]
        for n in range(0, n_layers):
            layer1.append(nn.Linear(h_dim, h_dim))
            if n < n_layers-1:
                layer1.append(nn.Softplus())
            else:
                if self.activation == 'softplus':
                    layer1.append(nn.Softplus())
                if self.activation == 'sigmoid':
                    layer1.append(nn.GroupNorm(1,h_dim))
                    layer1.append(nn.Sigmoid())
        self.layer1=nn.Sequential(*layer1)
        self.layer2=nn.Linear(h_dim, y_dim)
    def forward(self, x, c):
        #x.shape (N, 3)
        #c.shape (N, c_dim)
        #y_dim should be N*3
        c=c[0:1]
        c0=self.layer0(c)
        c1=self.layer1(c0)
        c2=self.layer2(c1)
        #print(x.shape, c2.shape)
        y=c2.view(x.shape[0], -1)
        return y
#%%
class MLP1b(nn.Module):
    def __init__(self, c_dim, h_dim, n_layers, y_dim, layer_norm=False, activation='softplus'):
        super().__init__()
        self.c_dim=c_dim
        self.h_dim=h_dim
        self.y_dim=y_dim
        self.layer_norm=layer_norm
        self.activation=activation
        if self.activation == 'softplus':
            self.layer0=nn.Sequential(nn.Linear(c_dim, h_dim),
                                      nn.Softplus())
        elif self.activation == 'relu':
            self.layer0=nn.Sequential(nn.Linear(c_dim, h_dim),
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
    def forward(self, x, c):
        #x.shape (N, 3)
        #c.shape (N, c_dim)
        #y_dim should be N*3 or N*9
        c=c[0:1]
        c0=self.layer0(c)
        c1=self.layer1(c0)
        c2=self.layer2(c1)
        #print(x.shape, c2.shape)
        y=c2.view(x.shape[0], -1)
        return y
#%%
class MLP1c(nn.Module):
    def __init__(self, c_dim, h_dim, n_layers, y_dim, activation='softplus'):
        super().__init__()
        self.c_dim=c_dim
        self.h_dim=h_dim
        self.y_dim=y_dim
        self.activation=activation
        if self.activation == 'softplus':
            self.layer0=nn.Sequential(nn.Linear(c_dim, h_dim),
                                      nn.Softplus())
        elif self.activation == 'relu':
            self.layer0=nn.Sequential(nn.Linear(c_dim, h_dim),
                                      nn.ReLU())
        else:
            raise ValueError('error')
        layer1=[]
        for n in range(0, n_layers):
            layer1.append(nn.Linear(h_dim, h_dim))
            if self.activation == 'softplus':
                layer1.append(nn.Softplus())
            elif self.activation == 'relu':
                layer1.append(nn.ReLU())
            else:
                raise ValueError('error')
            layer1.append(nn.GroupNorm(1, h_dim))
        self.layer1=nn.Sequential(*layer1)
        self.layer2=nn.Linear(h_dim, y_dim)
    def forward(self, x, c):
        #x.shape (N, 3)
        #c.shape (N, c_dim)
        #y_dim should be N*3 or N*9
        c=c[0:1]
        c0=self.layer0(c)
        c1=self.layer1(c0)
        c2=self.layer2(c1)
        #print(x.shape, c2.shape)
        y=c2.view(x.shape[0], -1)
        return y
#%%
class MLP2(nn.Module):
    def __init__(self, BaseNet, c_dim, h_dim, n_layers, y_dim, x_scale=1, alpha=1, gamma=1):
        super().__init__()
        self.BaseNet=BaseNet
        self.c_dim=c_dim
        self.h_dim=h_dim
        self.y_dim=y_dim
        self.x_scale=x_scale
        self.alpha=alpha
        self.gamma=gamma
        self.netA=eval(BaseNet+"("
                       +'x_dim='+str(c_dim)
                       +',h_dim='+str(h_dim)
                       +',n_layers='+str(n_layers)
                       +',x_scale='+str(x_scale)
                       +',alpha='+str(alpha)
                       +',gamma='+str(gamma)
                       +',y_dim='+str(y_dim)
                       +")"
                       )
    def forward(self, x, c):
        #x.shape (N, 3)
        #c.shape (N, c_dim)
        #y_dim should be N*3
        c=c[0:1]
        y=self.netA(c)
        y=y.view(x.shape[0], self.y_dim)
        return y
#%%
class Linear_encoder(nn.Module):
    def __init__(self, x_dim, y_dim, x_scale=1):
        super().__init__()
        self.x_dim=x_dim
        self.y_dim=y_dim
        self.x_scale=x_scale
        self.layer=nn.Linear(x_dim, y_dim)
    def forward(self, x, xx=None):
        #x.shape (N, 3)
        if self.x_scale != 1:
            x=x*self.x_scale
        x1=self.layer(x.view(1,-1))
        y=x1.view(1, -1).expand(x.shape[0], -1)
        return y
#%%
class Linear_decoder(nn.Module):
    def __init__(self, c_dim, y_dim):
        super().__init__()
        self.c_dim=c_dim
        self.y_dim=y_dim
        self.layer=nn.Linear(c_dim, y_dim)
    def forward(self, x, c):
        #x.shape (N, 3)
        #c.shape (N, c_dim)
        #y_dim should be N*3 or N*9
        c=c[0:1]
        y=self.layer(c)
        y=y.view(x.shape[0], -1)
        return y
#%%
class MLP_encoder1(nn.Module):
    def __init__(self, x_dim, h_dim, n_layers, y_dim, x_scale=1, activation='softplus'):
        super().__init__()
        self.x_dim=x_dim
        self.h_dim=h_dim
        self.y_dim=y_dim
        self.x_scale=x_scale
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
            elif self.activation == 'relu':
                layer1.append(nn.ReLU())
            elif self.activation == 'softplus_sigmoid':
                if n < n_layers-1:
                    layer1.append(nn.Softplus())
                else:
                    layer1.append(nn.GroupNorm(1,h_dim))
                    layer1.append(nn.Sigmoid())
            else:
                raise ValueError('error')
        self.layer1=nn.Sequential(*layer1)
        self.layer2=nn.Linear(h_dim, y_dim)
    def forward(self, x, xx=None):
        #x.shape (N, 3)
        if self.x_scale != 1:
            x=x*self.x_scale
        x0=self.layer0(x.view(1,-1))
        x1=self.layer1(x0)
        x2=self.layer2(x1)
        y=x2.view(1, -1).expand(x.shape[0], -1)
        return y
#%%
class MLP_encoder2(nn.Module):
    def __init__(self, BaseNet, x_dim, h_dim, n_layers, y_dim, x_scale, alpha, gamma):
        super().__init__()
        self.BaseNet=BaseNet
        self.x_dim=x_dim
        self.h_dim=h_dim
        self.y_dim=y_dim
        self.x_scale=x_scale
        self.alpha=alpha
        self.gamma=gamma
        self.netA=eval(BaseNet+"("
                       +'x_dim='+str(x_dim)
                       +',h_dim='+str(h_dim)
                       +',n_layers='+str(n_layers)
                       +',x_scale='+str(x_scale)
                       +',alpha='+str(alpha)
                       +',gamma='+str(gamma)
                       +',y_dim='+str(y_dim)
                       +")"
                       )
    def forward(self, x, xx=None):
        #x.shape (N, 3)
        y=self.netA(x.view(1,-1))
        y=y.view(1, -1).expand(x.shape[0], -1)
        return y
#%%
def classify_point(x, n_parts):
    #x.shape (N, 3)
    #p.shape (N, n_parts)
    N=x.shape[0]
    z=x[:,2] # in the range of -1 to 1
    z=z.view(N,1)
    a=torch.linspace(-1, 1, n_parts, dtype=x.dtype, device=x.device)
    a=a.view(1,n_parts)
    b=(0.3*2/n_parts)**2
    p=torch.exp(-0.5*(z-a)**2/b)
    p=p/p.sum(dim=1, keepdim=True)
    #print(z[0], p[0])
    return p
#%%
'''
x=torch.linspace(-1, 1, 100)
x=x.view(100,1).expand(100,3)
p=classify_point(x, 10)
#
x=x.numpy()
import matplotlib.pyplot as plt
import numpy as np
fig, ax =plt.subplots()
ax.plot(p)
'''
#%%
class PartNet(nn.Module):
    def __init__(self, n_parts, BaseNet, x_dim, c_dim, h_dim, n_layers, x_scale, alpha, gamma, y_dim):
        super().__init__()
        self.n_parts=n_parts
        self.BaseNet=BaseNet
        self.x_dim=x_dim
        self.c_dim=c_dim
        self.h_dim=h_dim
        self.n_layers=n_layers
        self.x_scale=x_scale
        self.alpha=alpha
        self.gamma=gamma
        self.y_dim=y_dim
        self.netA=nn.ModuleList()
        for n in range(0, n_parts):
            model=eval(BaseNet+"("
                       +'x_dim='+str(x_dim)
                       +',h_dim='+str(h_dim)
                       +',n_layers='+str(n_layers)
                       +',x_scale='+str(x_scale)
                       +',alpha='+str(alpha)
                       +',gamma='+str(gamma)
                       +',y_dim='+str(y_dim*h_dim)
                       +")"
                       )
            self.netA.append(model)
        self.netB=nn.Sequential(nn.Linear(c_dim, h_dim),
                                nn.Softplus(),
                                nn.Linear(h_dim, h_dim),
                                nn.Softplus(),
                                nn.Linear(h_dim, h_dim))

    def forward(self, x, c):
        p=classify_point(x, self.n_parts)
        y=predict(x, c, self.netA[0], self.netB, self.h_dim, self.y_dim)
        for n in range(1, self.n_parts):
            yn=predict(x, c, self.netA[n], self.netB, self.h_dim, self.y_dim)
            y=y+yn*p[:,n:(n+1)]
        return y
#%%
class NetA(nn.Module):
    def __init__(self, BaseNet, x_dim, c_dim, h_dim, n_layers, x_scale, alpha, gamma, y_dim):
        super().__init__()
        self.BaseNet=BaseNet
        self.x_dim=x_dim
        self.c_dim=c_dim
        self.h_dim=h_dim
        self.n_layers=n_layers
        self.x_scale=x_scale
        self.alpha=alpha
        self.gamma=gamma
        self.y_dim=y_dim
        self.netA=eval(BaseNet+"("
                       +'x_dim='+str(x_dim+c_dim)
                       +',h_dim='+str(h_dim)
                       +',n_layers='+str(n_layers)
                       +',x_scale='+str(x_scale)
                       +',alpha='+str(alpha)
                       +',gamma='+str(gamma)
                       +',y_dim='+str(y_dim)
                       +")"
                       )

    def forward(self, x, c):
        #x.shape (N, x_dim)
        #c.shape (N, c_dim)
        xc=torch.cat([x,c], dim=1)
        y=self.netA(xc)
        return y
#%%
from torch.nn import MultiheadAttention as MHA
from NNFEA_net_base import Sin
class TransEncoder1(nn.Module):
    def __init__(self, x_dim, h_dim, n_layers, num_heads, x_scale=1):
        super().__init__()
        self.x_dim=x_dim
        self.h_dim=h_dim
        self.n_layers=n_layers
        self.x_scale=x_scale
        self.netA1=nn.Sequential(Sin(x_dim, h_dim, alpha=1, is_first=True),
                                 nn.Linear(h_dim, h_dim))
        self.netA2=nn.Sequential(Sin(x_dim, h_dim, alpha=1, is_first=True),
                                 nn.Linear(h_dim, h_dim))
        self.netT=nn.ModuleList()
        for n in range(0, n_layers):
            self.netT.append(MHA(embed_dim=h_dim,
                                 num_heads=num_heads,
                                 dropout=0.0,
                                 bias=True,
                                 add_bias_kv=False,
                                 add_zero_attn=False,
                                 kdim=None,
                                 vdim=None,
                                 batch_first=True,
                                 device=None,
                                 dtype=None))
    def forward(self, x1, x2):
        #x1.shape (N, x_dim)
        #x2.shape (N, x_dim)
        if self.x_scale != 1:
            x1=x1*self.x_scale
            x2=x2*self.x_scale
        x1=self.netA1(x1)
        x2=self.netA1(x2)
        y=x1+x2 #y.shape (N,h_dim)
        y=y.view(1,y.shape[0],y.shape[1])
        for T in self.netT:
            y=T(y,y,y,need_weights=False)
            y=y[0]
        y=y.view(y.shape[1],y.shape[2])
        return y
#%%
class TransEncoder1a(nn.Module):
    def __init__(self, x_dim, h_dim, n_layers, num_heads, x_scale=1):
        super().__init__()
        self.x_dim=x_dim
        self.h_dim=h_dim
        self.n_layers=n_layers
        self.x_scale=x_scale
        self.netA=Sin(x_dim+x_dim, h_dim, alpha=1, is_first=True)
        self.netT=nn.ModuleList()
        for n in range(0, n_layers):
            self.netT.append(MHA(embed_dim=h_dim,
                                 num_heads=num_heads,
                                 dropout=0.0,
                                 bias=True,
                                 add_bias_kv=False,
                                 add_zero_attn=False,
                                 kdim=None,
                                 vdim=None,
                                 batch_first=True,
                                 device=None,
                                 dtype=None))
    def forward(self, x1, x2):
        #x1.shape (N, x_dim)
        #x2.shape (N, x_dim)
        if self.x_scale != 1:
            x1=x1*self.x_scale
            x2=x2*self.x_scale
        x=torch.cat([x1,x2], dim=1)
        x=self.netA(x)
        x=x.view(1,x.shape[0],x.shape[1])
        for T in self.netT:
            y=T(x,x,x,need_weights=False)
            y=y[0]
            x=y+x #residual connection
            #layer norm?
        y=y.view(y.shape[1],y.shape[2])
        return y
#%%
class TransEncoder1b(nn.Module):
    def __init__(self, x_dim, h_dim, n_layers, num_heads):
        super().__init__()
        self.x_dim=x_dim
        self.h_dim=h_dim
        self.n_layers=n_layers
        self.netA1=nn.Linear(x_dim, h_dim)
        self.netA2=nn.Sequential(Sin(x_dim, h_dim, alpha=1, is_first=True),
                                 nn.Linear(h_dim, h_dim))
        self.netT=nn.ModuleList()
        for n in range(0, n_layers):
            self.netT.append(MHA(embed_dim=h_dim,
                                 num_heads=num_heads,
                                 dropout=0.0,
                                 bias=True,
                                 add_bias_kv=False,
                                 add_zero_attn=False,
                                 kdim=None,
                                 vdim=None,
                                 batch_first=True,
                                 device=None,
                                 dtype=None))
    def forward(self, x1, x2):
        #x1.shape (N, x_dim)
        #x2.shape (N, x_dim)
        #x1=x1*0.01
        x1=self.netA1(x1)
        x2=self.netA1(x2)
        y=x1+x2 #y.shape (N,h_dim)
        y=y.view(1,y.shape[0],y.shape[1])
        for T in self.netT:
            y=T(y,y,y,need_weights=False)
            y=y[0]
        y=y.view(y.shape[1],y.shape[2])
        return y
#%%
class TransEncoder2(nn.Module):
    def __init__(self, x_dim, h_dim, n_layers, num_heads):
        super().__init__()
        self.x_dim=x_dim
        self.h_dim=h_dim
        self.n_layers=n_layers
        self.netA=eval("BaseNet5b("
                       +'x_dim='+str(x_dim+x_dim)
                       +',h_dim=128'
                       +',n_layers=2'
                       +',x_scale=1'
                       +',alpha=0.1'
                       +',gamma=1'
                       +',y_dim='+str(h_dim)
                       +")"
                       )
        self.netT=nn.ModuleList()
        for n in range(0, n_layers):
            self.netT.append(MHA(embed_dim=h_dim,
                                 num_heads=num_heads,
                                 dropout=0.0,
                                 bias=True,
                                 add_bias_kv=False,
                                 add_zero_attn=False,
                                 kdim=None,
                                 vdim=None,
                                 batch_first=True,
                                 device=None,
                                 dtype=None))
    def forward(self, x1, x2):
        #x1.shape (N, x_dim)
        #x2.shape (N, x_dim)
        x=torch.cat([x1,x2], dim=1)
        x=self.netA(x)
        x=x.view(1,x.shape[0],x.shape[1])
        for T in self.netT:
            y=T(x,x,x,need_weights=False)
            y=y[0]
            x=y+x #residual connection
            y=x
            #layer norm?
        y=y.view(y.shape[1],y.shape[2])
        return y
#%%
class TransDecoder1(nn.Module):
    def __init__(self, c_dim, y_dim):
        super().__init__()
        self.c_dim=c_dim
        self.y_dim=y_dim
        self.layer=nn.Linear(c_dim, y_dim)
    def forward(self, x, c):
        #x.shape (N, 3)
        #c.shape (N, c_dim)
        #y_dim should be 3 (shape) or 9(stress)
        y=self.layer(c)
        return y
#%%
class TransDecoder2(nn.Module):
    def __init__(self, BaseNet, x_dim, c_dim, h_dim, k_dim, n_layers, y_dim, activation):
        super().__init__()
        self.BaseNet=BaseNet
        self.x_dim=x_dim
        self.h_dim=h_dim
        self.k_dim=k_dim
        self.n_layers=n_layers
        self.y_dim=y_dim
        self.activation=activation
        self.netQ=eval(BaseNet+"("
                       +'x_dim='+str(x_dim)
                       +',h_dim='+str(h_dim)
                       +',n_layers='+str(n_layers)
                       +',x_scale=1'
                       +',alpha=1'
                       +',gamma=1'
                       +',y_dim='+str(k_dim)
                       +")"
                       )
        self.netK=eval(BaseNet+"("
                       +'x_dim='+str(c_dim)
                       +',h_dim='+str(h_dim)
                       +',n_layers='+str(n_layers)
                       +',x_scale=1'
                       +',alpha=1'
                       +',gamma=1'
                       +',y_dim='+str(k_dim)
                       +")"
                       )
        self.netV=eval(BaseNet+"("
                       +'x_dim='+str(c_dim)
                       +',h_dim='+str(h_dim)
                       +',n_layers='+str(n_layers)
                       +',x_scale=1'
                       +',alpha=1'
                       +',gamma=1'
                       +',y_dim='+str(y_dim)
                       +")"
                       )
    def forward(self, x, c):
        Q=self.netQ(x) #Q.shape (N, k_dim)
        K=self.netK(c) #K.shape (M, k_dim)
        V=self.netV(c) #V.shape (M, y_dim)
        A=torch.matmul(Q, K.permute(1,0)) # (N, M)
        if self.activation=='sigmoid':
            A=nnF.sigmoid(A)
        elif self.activation=='softplus':
            A=nnF.softplus(A)
        elif self.activation=='softmax':
            A=nnF.softmax(A, dim=1)
        y=torch.matmul(A, V) #shape (N, y_dim)
        return y
#%%
class Encoder0(nn.Module):
    def __init__(self, BaseNet, x_dim, h_dim, n_layers, x_scale, alpha, gamma, y_dim):
        super().__init__()
        self.BaseNet=BaseNet
        self.x_dim=x_dim
        self.h_dim=h_dim
        self.n_layers=n_layers
        self.x_scale=x_scale
        self.alpha=alpha
        self.gamma=gamma
        self.y_dim=y_dim
        self.netA1=nn.Sequential(Sin(x_dim, h_dim, alpha=alpha, is_first=True),
                                 nn.Linear(h_dim, h_dim))
        self.netA2=nn.Sequential(Sin(x_dim, h_dim, alpha=alpha, is_first=True),
                                 nn.Linear(h_dim, h_dim))
        self.netA3=eval(BaseNet+"("
                       +'x_dim='+str(h_dim)
                       +',h_dim='+str(h_dim)
                       +',n_layers='+str(n_layers)
                       +',x_scale=1'
                       +',alpha=1'
                       +',gamma='+str(gamma)
                       +',y_dim='+str(y_dim)
                       +")"
                       )
    def forward(self, x1, x2):
        #x1.shape (N, x_dim)
        #x2.shape (N, x_dim)
        if self.x_scale != 1:
            x1=x1*self.x_scale
            x2=x2*self.x_scale
        y1=self.netA1(x1)
        y2=self.netA2(x2)
        y=self.netA3(y1+y2)
        #y.shape (N, y_dim)
        y=y.mean(dim=0, keepdim=True)
        y=y.expand(x1.shape[0], self.y_dim)
        return y
#%%
class Encoder0a(nn.Module):
    def __init__(self, BaseNet, x_dim, h_dim, n_layers, x_scale, alpha, gamma, y_dim):
        super().__init__()
        self.BaseNet=BaseNet
        self.x_dim=x_dim
        self.h_dim=h_dim
        self.n_layers=n_layers
        self.x_scale=x_scale
        self.alpha=alpha
        self.gamma=gamma
        self.y_dim=y_dim
        self.netA=eval(BaseNet+"("
                       +'x_dim='+str(x_dim)
                       +',h_dim='+str(h_dim)
                       +',n_layers='+str(n_layers)
                       +',x_scale=1'
                       +',alpha='+str(alpha)
                       +',gamma='+str(gamma)
                       +',y_dim='+str(y_dim)
                       +")"
                       )
    def forward(self, x1, x2):
        #x1.shape (N, x_dim)
        #x2.shape (N, x_dim)
        if self.x_scale != 1:
            x1=x1*self.x_scale
            x2=x2*self.x_scale
        y=self.netA(x1)
        #y.shape (N, y_dim)
        y=y.mean(dim=0, keepdim=True)
        y=y.expand(x1.shape[0], self.y_dim)
        return y
#%%
class Encoder1(nn.Module):
    def __init__(self, BaseNet, x_dim, h_dim, n_layers, x_scale, alpha, gamma, y_dim):
        super().__init__()
        self.BaseNet=BaseNet
        self.x_dim=x_dim
        self.h_dim=h_dim
        self.n_layers=n_layers
        self.x_scale=x_scale
        self.alpha=alpha
        self.gamma=gamma
        self.y_dim=y_dim
        self.netA=eval(BaseNet+"("
                       +'x_dim='+str(x_dim+x_dim)
                       +',h_dim='+str(h_dim)
                       +',n_layers='+str(n_layers)
                       +',x_scale=1'
                       +',alpha='+str(alpha)
                       +',gamma='+str(gamma)
                       +',y_dim='+str(y_dim)
                       +")"
                       )
    def forward(self, x1, x2):
        #x1.shape (N, x_dim)
        #x2.shape (N, x_dim)
        if self.x_scale != 1:
            x1=x1*self.x_scale
            x2=x2*self.x_scale
        x=torch.cat([x1,x2], dim=1)
        y=self.netA(x)
        #y.shape (N, y_dim)
        y=y.mean(dim=0, keepdim=True)
        y=y.expand(x1.shape[0], self.y_dim)
        return y
#%%
class Encoder2(nn.Module):
    def __init__(self, BaseNet, x_dim, h_dim, n_layers, x_scale, alpha, gamma, y_dim):
        super().__init__()
        self.BaseNet=BaseNet
        self.x_dim=x_dim
        self.h_dim=h_dim
        self.n_layers=n_layers
        self.x_scale=x_scale
        self.alpha=alpha
        self.gamma=gamma
        self.y_dim=y_dim
        self.netA=eval(BaseNet+"("
                       +'x_dim='+str(x_dim+x_dim)
                       +',h_dim='+str(h_dim)
                       +',n_layers='+str(n_layers)
                       +',x_scale=1'
                       +',alpha='+str(alpha)
                       +',gamma='+str(gamma)
                       +',y_dim='+str(x_dim*y_dim)
                       +")"
                       )
    def forward(self, x1, x2):
        #x1.shape (N, x_dim)
        #x2.shape (N, x_dim)
        if self.x_scale != 1:
            x1=x1*self.x_scale
            x2=x2*self.x_scale
        x=torch.cat([x1,x2], dim=1)
        y1=self.netA(x)
        #y.shape (N, y_dimx_dim)
        y1=y1.view(x1.shape[0], self.x_dim, self.y_dim)
        x1=x1.view(x1.shape[0], self.x_dim, 1)
        y=(y1*x1).mean(dim=(0,1))
        y=y.view(1,-1)
        y=y.expand(x1.shape[0], self.y_dim)
        return y
#%%
class Encoder3(nn.Module):
    def __init__(self, BaseNet, x_dim, h_dim, n_layers, x_scale, alpha, gamma, y_dim):
        super().__init__()
        self.BaseNet=BaseNet
        self.x_dim=x_dim
        self.h_dim=h_dim
        self.n_layers=n_layers
        self.x_scale=x_scale
        self.alpha=alpha
        self.gamma=gamma
        self.y_dim=y_dim
        self.netA=eval(BaseNet+"("
                       +'x_dim='+str(x_dim)
                       +',h_dim='+str(h_dim)
                       +',n_layers='+str(n_layers)
                       +',x_scale=1'
                       +',alpha='+str(alpha)
                       +',gamma='+str(gamma)
                       +',y_dim='+str(x_dim*y_dim)
                       +")"
                       )
    def forward(self, x1, x2, N=None):
        #x1.shape (N, x_dim)
        #x2.shape (N, x_dim)
        if self.x_scale != 1:
            x1=x1*self.x_scale
            x2=x2*self.x_scale
        y1=self.netA(x2)
        #y.shape (N, y_dimx_dim)
        y1=y1.view(x1.shape[0], self.x_dim, self.y_dim)
        x1=x1.view(x1.shape[0], self.x_dim, 1)
        y=(y1*x1).mean(dim=(0,1))
        y=y.view(1,-1)
        if N is None:
            y=y.expand(x1.shape[0], self.y_dim)
        else:
            y=y.expand(N, self.y_dim)
        return y
#%%
class Decoder1(nn.Module):
    def __init__(self, BaseNet, x_dim, c_dim, h_dim, n_layers, x_scale, y_dim, activation='sigmoid'):
        super().__init__()
        self.BaseNet=BaseNet
        self.x_dim=x_dim
        self.c_dim=c_dim
        self.h_dim=h_dim
        self.n_layers=n_layers
        self.x_scale=x_scale
        self.y_dim=y_dim
        self.activation=activation
        self.netQ=eval(BaseNet+"("
                       +'x_dim='+str(x_dim)
                       +',h_dim='+str(h_dim)
                       +',n_layers='+str(n_layers)
                       +',x_scale=1'
                       +',alpha=1'
                       +',gamma=1'
                       +',y_dim='+str(h_dim)
                       +")"
                       )
        self.netK=eval(BaseNet+"("
                       +'x_dim='+str(c_dim)
                       +',h_dim='+str(h_dim)
                       +',n_layers='+str(n_layers)
                       +',x_scale=1'
                       +',alpha=1'
                       +',gamma=1'
                       +',y_dim='+str(h_dim*h_dim)
                       +")"
                       )
        self.netV=eval(BaseNet+"("
                       +'x_dim='+str(c_dim)
                       +',h_dim='+str(h_dim)
                       +',n_layers='+str(n_layers)
                       +',x_scale=1'
                       +',alpha=1'
                       +',gamma=1'
                       +',y_dim='+str(h_dim*y_dim)
                       +")"
                       )
    def forward(self, x, c):
        #x.shape (N, x_dim)
        #c.shape (N, c_dim)
        if self.x_scale != 1:
            x=x*self.x_scale
        Q=self.netQ(x) #Q.shape (N, h_dim)
        K=self.netK(c[0:1]) #K.shape (1,h_dim*h_dim)
        K=K.view(self.h_dim, self.h_dim)
        V=self.netV(c[0:1]) #V.shape (1,h_dim*y_dim)
        V=V.view(self.h_dim, self.y_dim)
        A=torch.matmul(Q, K) # (N, h_dim)
        if self.activation=='sigmoid':
            A=nnF.sigmoid(A)
        elif self.activation=='softplus':
            A=nnF.softplus(A)
        elif self.activation=='softmax':
            A=nnF.softmax(A, dim=1)

        y=torch.matmul(A, V) #shape (N, y_dim)
        return y

#%%
class Decoder2(nn.Module):
    def __init__(self, BaseNet, x_dim, c_dim, h_dim, k_dim, n_layers, x_scale, y_dim, activation='sigmoid'):
        super().__init__()
        self.BaseNet=BaseNet
        self.x_dim=x_dim
        self.c_dim=c_dim
        self.h_dim=h_dim
        self.k_dim=k_dim
        self.n_layers=n_layers
        self.x_scale=x_scale
        self.y_dim=y_dim
        self.activation=activation
        self.netQ=eval(BaseNet+"("
                       +'x_dim='+str(x_dim)
                       +',h_dim='+str(h_dim)
                       +',n_layers='+str(n_layers)
                       +',x_scale=1'
                       +',alpha=1'
                       +',gamma=1'
                       +',y_dim='+str(k_dim)
                       +")"
                       )
        self.netK=eval(BaseNet+"("
                       +'x_dim='+str(c_dim)
                       +',h_dim='+str(h_dim)
                       +',n_layers='+str(n_layers)
                       +',x_scale=1'
                       +',alpha=1'
                       +',gamma=1'
                       +',y_dim='+str(h_dim*k_dim)
                       +")"
                       )
        self.netV=eval(BaseNet+"("
                       +'x_dim='+str(c_dim)
                       +',h_dim='+str(h_dim)
                       +',n_layers='+str(n_layers)
                       +',x_scale=1'
                       +',alpha=1'
                       +',gamma=1'
                       +',y_dim='+str(h_dim*y_dim)
                       +")"
                       )
    def forward(self, x, c):
        #x.shape (N, x_dim)
        #c.shape (N, c_dim)
        if self.x_scale != 1:
            x=x*self.x_scale
        Q=self.netQ(x) #Q.shape (N, k_dim)
        K=self.netK(c[0:1]) #K.shape (1,h_dim*k_dim)
        K=K.view(self.k_dim, self.h_dim)
        V=self.netV(c[0:1]) #V.shape (1,h_dim*y_dim)
        V=V.view(self.h_dim, self.y_dim)
        A=torch.matmul(Q, K) # (N, h_dim)
        if self.activation=='sigmoid':
            A=nnF.sigmoid(A)
        elif self.activation=='softplus':
            A=nnF.softplus(A)
        elif self.activation=='softmax':
            A=nnF.softmax(A, dim=1)

        y=torch.matmul(A, V) #shape (N, y_dim)
        return y
