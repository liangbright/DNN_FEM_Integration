import torch
import torch.nn as nn
from NNFEA_net_x_c import Net1, Encoder3, MLP1b
#%%
class NetXCM1(nn.Module):
    def __init__(self, n_models, encoder, decoder):
        super().__init__()
        self.n_models=n_models
        self.encoder=eval(encoder)#only need one shape encoder
        self.decoder=nn.ModuleList()
        for n in range(0, n_models):
            self.decoder.append(eval(decoder))
        self.netW=nn.Sequential(nn.Linear(5,128),
                                nn.Softplus(),
                                nn.Linear(128,128),
                                nn.Softplus(),
                                nn.Linear(128,n_models),
                                nn.Softmax(dim=-1))

    def forward(self, x, x_mean, mat):
        m=mat.view(1,5)
        c=self.encoder(x-x_mean, x_mean, N=1)
        #c.shape (1, c_dim)
        #print("c", c[0])
        #print("m", m[0])
        cm=torch.cat([c,m], dim=1)
        #cm=cm.expand(x.shape[0],cm.shape[1])
        u_pred_all=[]
        for n in range(0, self.n_models):
            u_pred=self.decoder[n](x_mean, cm)
            #u_pred.shape (N,3)
            u_pred=u_pred.view(1,u_pred.shape[0],3)
            u_pred_all.append(u_pred)
        #u_pred_all.shape (n_models, N, 3)
        if self.n_models > 1:
            W=self.netW(m) #W.shape (1, n_models)
            W=W.view(-1,1,1)
            u_pred_all=torch.cat(u_pred_all, dim=0)
            u_pred_out=(W*u_pred_all).sum(dim=0)
            #print(W)
        else:
            u_pred_out=u_pred.view(-1,3)
        return u_pred_out
#%%
class NetXCM1A(nn.Module):
    def __init__(self, n_models, encoder, decoder):
        super().__init__()
        self.n_models=n_models
        self.encoder=eval(encoder)#only need one shape encoder
        self.decoder=nn.ModuleList()
        for n in range(0, n_models):
            self.decoder.append(eval(decoder))
        self.netM=nn.Sequential(nn.Linear(5,128),
                                nn.Softplus(),
                                nn.Linear(128,128),
                                nn.Softplus(),
                                nn.Linear(128,5))
        self.netW=nn.Sequential(nn.Linear(5,128),
                                nn.Softplus(),
                                nn.Linear(128,128),
                                nn.Softplus(),
                                nn.Linear(128,n_models),
                                nn.Softmax(dim=-1))

    def forward(self, x, x_mean, mat):
        mat=mat.view(1,5)
        m=self.netM(mat)
        c=self.encoder(x-x_mean, x_mean, N=1)
        #c.shape (1, c_dim)
        #print("c", c[0])
        #print("m", m[0])
        cm=torch.cat([c,m], dim=1)
        #cm=cm.expand(x.shape[0],cm.shape[1])
        u_pred_all=[]
        for n in range(0, self.n_models):
            u_pred=self.decoder[n](x_mean, cm)
            #u_pred.shape (N,3)
            u_pred=u_pred.view(1,u_pred.shape[0],3)
            u_pred_all.append(u_pred)
        #u_pred_all.shape (n_models, N, 3)
        if self.n_models > 1:
            W=self.netW(mat) #W.shape (1, n_models)
            W=W.view(-1,1,1)
            u_pred_all=torch.cat(u_pred_all, dim=0)
            u_pred_out=(W*u_pred_all).sum(dim=0)
            #print(W)
        else:
            u_pred_out=u_pred.view(-1,3)
        return u_pred_out
#%%
class NetXCM2(nn.Module):
    def __init__(self, n_models, encoder, decoder):
        super().__init__()
        self.n_models=n_models
        self.encoder=eval(encoder)#only need one shape encoder
        self.decoder=nn.ModuleList()
        for n in range(0, n_models):
            self.decoder.append(eval(decoder))

    def forward(self, x, x_mean, mat):
        m=mat.view(1,5)
        c=self.encoder(x-x_mean, x_mean, N=1)
        #c.shape (1, c_dim)
        cm=torch.cat([c,m], dim=1)
        #cm=cm.expand(x.shape[0],cm.shape[1])
        u_pred_all=[]
        for n in range(0, self.n_models):
            u_pred=self.decoder[n](x_mean, cm)
            #u_pred.shape (N,3)
            u_pred=u_pred.view(1,u_pred.shape[0],3)
            u_pred_all.append(u_pred)
        #u_pred_all.shape (n_models, N, 3)
        if self.n_models > 1:
            u_pred_all=torch.cat(u_pred_all, dim=0)
            u_pred_out=u_pred_all.mean(dim=0)
        else:
            u_pred_out=u_pred.view(-1,3)
        return u_pred_out
#%%
class NetXCM2A(nn.Module):
    def __init__(self, n_models, encoder, decoder):
        super().__init__()
        self.n_models=n_models
        self.encoder=eval(encoder)#only need one shape encoder
        self.decoder=nn.ModuleList()
        for n in range(0, n_models):
            self.decoder.append(eval(decoder))
        self.netM=nn.Sequential(nn.Linear(5,128),
                                nn.Softplus(),
                                nn.Linear(128,128),
                                nn.Softplus(),
                                nn.Linear(128,5))

    def forward(self, x, x_mean, mat):
        mat=mat.view(1,5)
        m=self.netM(mat)
        c=self.encoder(x-x_mean, x_mean, N=1)
        #c.shape (1, c_dim)
        cm=torch.cat([c,m], dim=1)
        #cm=cm.expand(x.shape[0],cm.shape[1])
        u_pred_all=[]
        for n in range(0, self.n_models):
            u_pred=self.decoder[n](x_mean, cm)
            #u_pred.shape (N,3)
            u_pred=u_pred.view(1,u_pred.shape[0],3)
            u_pred_all.append(u_pred)
        #u_pred_all.shape (n_models, N, 3)
        if self.n_models > 1:
            u_pred_all=torch.cat(u_pred_all, dim=0)
            u_pred_out=u_pred_all.mean(dim=0)
        else:
            u_pred_out=u_pred.view(-1,3)
        return u_pred_out
#%%
class NetXCM2A1(nn.Module):
    def __init__(self, n_models, encoder, decoder):
        super().__init__()
        self.n_models=n_models
        self.encoder=eval(encoder)#only need one shape encoder
        self.decoder=nn.ModuleList()
        for n in range(0, n_models):
            self.decoder.append(eval(decoder))
        self.netM=nn.Linear(5,5)

    def forward(self, x, x_mean, mat):
        mat=mat.view(1,5)
        m=self.netM(mat)
        c=self.encoder(x-x_mean, x_mean, N=1)
        #c.shape (1, c_dim)
        cm=torch.cat([c,m], dim=1)
        #cm=cm.expand(x.shape[0],cm.shape[1])
        u_pred_all=[]
        for n in range(0, self.n_models):
            u_pred=self.decoder[n](x_mean, cm)
            #u_pred.shape (N,3)
            u_pred=u_pred.view(1,u_pred.shape[0],3)
            u_pred_all.append(u_pred)
        #u_pred_all.shape (n_models, N, 3)
        if self.n_models > 1:
            u_pred_all=torch.cat(u_pred_all, dim=0)
            u_pred_out=u_pred_all.mean(dim=0)
        else:
            u_pred_out=u_pred.view(-1,3)
        return u_pred_out
#%%
'''

#'''