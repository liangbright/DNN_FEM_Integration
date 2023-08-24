import torch
import torch.nn as nn
import torch.nn.functional as nnF
from NNFEA_net_base import LinearRes, Sin, preprocess
from NNFEA_net_base import BaseNet0, BaseNet1, BaseNet1a, BaseNet2, BaseNet2a, BaseNet2b, BaseNet3, BaseNet3b
from NNFEA_net_base import BaseNet4, BaseNet4a, BaseNet4b, BaseNet5a, BaseNet5b
from NNFEA_net_base import BaseNet6a, BaseNet6b
#%%
class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim=embed_dim
        self.num_heads=num_heads
        self.MHA=nn.MultiheadAttention(embed_dim, num_heads,
                                       dropout=0.0,
                                       bias=True,
                                       add_bias_kv=False,
                                       add_zero_attn=False,
                                       kdim=None,
                                       vdim=None,
                                       batch_first=True,
                                       device=None,
                                       dtype=None)
        self.MLP=nn.Sequential(nn.Linear(embed_dim, 2*embed_dim),
                               nn.GELU(),
                               nn.Linear(2*embed_dim, embed_dim)
                               )
        self.Norm1q=nn.LayerNorm(embed_dim)
        self.Norm1k=nn.LayerNorm(embed_dim)
        self.Norm2=nn.LayerNorm(embed_dim)
    def forward(self, q, k, v):
        #x.shape is (B, L, E), E=embed_dim
        #usually, the shape of a feature map is (B,C,H,W)
        #assume permute and reshape/view have been applied, then L=H*W and C=E=embed_dim
        #--------------------------------------------
        x=q+self.MHA(self.Norm1q(q), self.Norm1k(k), v)[0]#(B,L,E)
        #add&norm
        y=x+self.MLP(self.Norm2(x))
        return y #(B,L,E)
#%%
class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim=embed_dim
        self.num_heads=num_heads
        self.MHA=nn.MultiheadAttention(embed_dim, num_heads,
                                       dropout=0.0,
                                       bias=True,
                                       add_bias_kv=False,
                                       add_zero_attn=False,
                                       kdim=None,
                                       vdim=None,
                                       batch_first=True,
                                       device=None,
                                       dtype=None)
        self.MLP=nn.Sequential(nn.Linear(embed_dim, 2*embed_dim),
                               nn.GELU(),
                               nn.Linear(2*embed_dim, embed_dim)
                               )
        self.Norm1=nn.LayerNorm(embed_dim)
        self.Norm2=nn.LayerNorm(embed_dim)
    def forward(self, x):
        #x.shape is (B, L, E), E=embed_dim
        #usually, the shape of a feature map is (B,C,H,W)
        #assume permute and reshape/view have been applied, then L=H*W and C=E=embed_dim
        #--------------------------------------------
        x1=self.Norm1(x)
        x2=x+self.MHA(x1,x1,x1)[0]#(B,L,E)
        #add&norm
        y=x2+self.MLP(self.Norm2(x2))
        return y #(B,L,E)
#%%
class TransEncoder4(nn.Module):
    def __init__(self, x_layers, h_dim, n_layers, num_heads, alpha=1):
        super().__init__()
        self.x_layers=x_layers
        self.h_dim=h_dim
        self.n_layers=n_layers

        #not good compared to a single linear
        #self.patch_layer=nn.Sequential(Sin(50*3*x_layers, h_dim, alpha, True),
        #                               Sin(h_dim, h_dim, 1, False),
        #                               nn.Linear(h_dim, h_dim))
        #self.patch_layer=nn.Sequential(Sin(50*3*x_layers, h_dim, alpha, True),
        #                               nn.Linear(h_dim, h_dim))

        self.patch_layer=nn.Linear(50*3*x_layers, h_dim)

        self.netT=nn.Sequential()
        for n in range(0, n_layers):
            self.netT.append(SelfAttentionBlock(h_dim, num_heads))

        self.out=nn.Sequential(nn.LayerNorm(h_dim), #better to use norm
                               nn.Linear(h_dim, 50*3*x_layers))

        #not good, compared to sin-sin-linear
        #self.pos=nn.Parameter(0.02*torch.randn(100, h_dim))

        #not good compared to sin-sin-linear
        #self.pos=nn.Linear(3, h_dim)

        self.pos=nn.Sequential(Sin(3, h_dim, alpha, True),
                               Sin(h_dim, h_dim, 1, False),
                               nn.Linear(h_dim, h_dim))


    def get_patch(self, x):
        #x.shape  (N, x_dim), N=5000*2
        y=x.view(-1,5000,3)
        x_layers=y.shape[0]
        y=y.reshape(x_layers,100,50,3)
        y=y.permute(1,0,2,3) #(100,x_layers,50,3)
        y=y.reshape(1,100,-1)
        return y

    def un_patch(sefl, y):
        #y.shape (1,100,-1)
        y=y.view(100,-1).view(100,-1,50,3)
        y=y.permute(1,0,2,3)
        y=y.reshape(-1,3)
        return y

    def forward(self, x1, x2):
        #x1.shape (N, x_dim)  x-meanshape
        #x2.shape (N, x_dim)   meanshape
        #---------------
        #"Patch"
        #print('0', x1.shape)
        x1=self.get_patch(x1)
        #print('1', x.shape)
        x1=self.patch_layer(x1) #(100, h_dim)
        #---------------
        if torch.is_tensor(self.pos):
            x2=self.pos
        else:
            x2=self.get_patch(x2).reshape(100,-1,3).mean(dim=1)
            x2=self.pos(x2)
        #---------------
        #pos embedding
        x=x1+x2
        #print(x.shape)
        #-------------
        x=self.netT(x)
        y=self.out(x) #(100, 50*x_layers)
        #print('2', y.shape)
        y=self.un_patch(y)
        return y
#%%
class TransEncoder4A(nn.Module):
    def __init__(self, x_layers, h_dim, n_layers, num_heads, alpha=1):
        super().__init__()
        self.x_layers=x_layers
        self.h_dim=h_dim
        self.n_layers=n_layers

        self.netA=Sin(6, h_dim, alpha=1, is_first=True)

        self.patch_layer=nn.Linear(50*x_layers*6, h_dim)

        self.netT=nn.Sequential()
        for n in range(0, n_layers):
            self.netT.append(SelfAttentionBlock(h_dim, num_heads))

        self.out=nn.Sequential(nn.LayerNorm(h_dim), #better to use norm
                               nn.Linear(h_dim, 50*3*x_layers))

        #not good, compared to sin-sin-linear
        #self.pos=nn.Parameter(0.02*torch.randn(100, h_dim))

        #not good compared to sin-sin-linear
        #self.pos=nn.Linear(3, h_dim)

    def get_patch(self, x):
        #x.shape  (N, x_dim), N=5000*2
        y=x.view(-1,5000,3)
        x_layers=y.shape[0]
        y=y.reshape(x_layers,100,50,3)
        y=y.permute(1,0,2,3) #(100,x_layers,50,3)
        y=y.reshape(1,100,-1)
        return y

    def un_patch(sefl, y):
        #y.shape (1,100,-1)
        y=y.view(100,-1).view(100,-1,50,3)
        y=y.permute(1,0,2,3)
        y=y.reshape(-1,3)
        return y

    def forward(self, x1, x2):
        #x1.shape (N, x_dim)  x-meanshape
        #x2.shape (N, x_dim)   meanshape
        #---------------
        #"Patch"
        #print('0', x1.shape)
        x=torch.cat([x1,x2], dim=1)
        #x=self.netA(x)
        x=self.get_patch(x, self.x_layers)
        #print('1', x.shape)
        x=self.patch_layer(x) #(100, h_dim)
        #-------------
        x=x.view(1,x.shape[0],x.shape[1])
        x=self.netT(x)
        y=self.out(x) #(100, 50*x_layers)
        #print('2', y.shape)
        y=y.view(-1,3)
        return y
#%%
class TransDecoder4(nn.Module):
    def __init__(self,):
        super().__init__()
        self.p=nn.Parameter(torch.zeros(1))
    def forward(self, x, c):
        #x: meanshape
        #c: code
        return c
#%%
class TransEncoder5(nn.Module):
    def __init__(self, x_layers, h_dim, n_layers, num_heads, alpha=1):
        super().__init__()
        self.x_layers=x_layers
        self.h_dim=h_dim
        self.n_layers=n_layers

        self.patch_layer=nn.Linear(50*3*x_layers, h_dim)

        self.netT=nn.Sequential()
        for n in range(0, n_layers):
            self.netT.append(SelfAttentionBlock(h_dim, num_heads))

        self.out=nn.Linear(h_dim, h_dim)

        self.pos=nn.Sequential(Sin(3, h_dim, alpha, True),
                               Sin(h_dim, h_dim, 1, False),
                               nn.Linear(h_dim, h_dim))

    def get_patch(self, x):
        #x.shape  (N, x_dim), N=5000*2
        y=x.view(-1,5000,3)
        x_layers=y.shape[0]
        y=y.reshape(x_layers,100,50,3)
        y=y.permute(1,0,2,3) #(100,x_layers,50,3)
        y=y.reshape(1,100,-1)
        return y

    def forward(self, x1, x2):
        #x1.shape (N, x_dim)  x-meanshape
        #x2.shape (N, x_dim)   meanshape
        #---------------
        #"Patch"
        #print('0', x1.shape)
        x=self.get_patch(x1)
        #print('1', x.shape)
        x=self.patch_layer(x) #(100, h_dim)
        #---------------
        #pos embedding
        if torch.is_tensor(self.pos):
            x2=self.pos
        else:
            x2=self.get_patch(x2).reshape(100,-1,3).mean(dim=1)
            x2=self.pos(x2)
        #-------------
        x=self.netT(x)
        y=self.out(x) #(100, h_dim)
        return y
#%%
class TransDecoder5(nn.Module):
    def __init__(self, x_layers, h_dim, n_layers, num_heads):
        super().__init__()
        self.x_layers=x_layers
        self.h_dim=h_dim
        self.n_layers=n_layers
        self.netQ=eval("BaseNet5b("
                       +'x_dim=3'
                       +',h_dim=128'
                       +',n_layers=2'
                       +',x_scale=1'
                       +',alpha=1'
                       +',gamma=1'
                       +',y_dim='+str(h_dim)
                       +")"
                       )
        self.attnA=nn.ModuleList()
        self.attnB=nn.ModuleList()
        for n in range(0, n_layers):
            self.attnA.append(AttentionBlock(h_dim, num_heads))
            self.attnB.append(SelfAttentionBlock(h_dim, num_heads))

        self.out=nn.Linear(h_dim, 3)

    def forward(self, x, c):
        #x: meanshape
        #c: code
        #print(c.shape)
        Q=self.netQ(x) #Q.shape (N, h_dim)
        Q=Q.view(1,Q.shape[0],Q.shape[1])
        V=K=c
        for n in range(self. n_layers):
            A=self.attnA[n](Q,K,V)
            B=self.attnB[n](A)
            Q=B
        y=self.out(B)
        y=y.view(-1,3)
        return y
#%%
class TransDecoder5A(nn.Module):
    def __init__(self, x_layers, h_dim, n_layers, num_heads):
        super().__init__()
        self.x_layers=x_layers
        self.h_dim=h_dim
        self.n_layers=n_layers
        self.netQ=nn.Linear(3, h_dim)
        self.attnA=nn.ModuleList()
        self.attnB=nn.ModuleList()
        for n in range(0, n_layers):
            self.attnA.append(AttentionBlock(h_dim, num_heads))
            self.attnB.append(SelfAttentionBlock(h_dim, num_heads))

        self.out=nn.Linear(h_dim, 3)

    def forward(self, x, c):
        #x: meanshape
        #c: code
        #print(c.shape)
        Q=self.netQ(x) #Q.shape (N, h_dim)
        Q=Q.view(1,Q.shape[0],Q.shape[1])
        V=K=c
        for n in range(self. n_layers):
            A=self.attnA[n](Q,K,V)
            B=self.attnB[n](A)
            Q=B
        y=self.out(B)
        y=y.view(-1,3)
        return y