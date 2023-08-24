import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import numpy as np
import torch.nn.functional as nnF
import torch.nn as nn
import torch
#%%
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        #embed_dim is useful
        embed_dim=max(in_channels, out_channels)
        
        self.conv=nn.Sequential(nn.Conv1d(in_channels, embed_dim, kernel_size, stride, padding),
                                nn.GroupNorm(1, embed_dim),
                                nn.LeakyReLU(inplace=True),
                                nn.Conv1d(embed_dim, out_channels, kernel_size=3, stride=1, padding=1))
        self.res_path=nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        
        self.out=nn.Sequential(nn.GroupNorm(1, out_channels),
                               nn.LeakyReLU(inplace=True))
                
    def forward(self, x):
        #x.shape (B,C,H,W), C=in_channels        
        y=self.res_path(x)+self.conv(x) #(B,out_channels,H_new,W_new)
        y=self.out(y)
        return y
#%%
class MergeLayer(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, scale_factor):
        super().__init__()        
        self.up = nn.Upsample(scale_factor=scale_factor)
        self.projector1=nn.Conv1d(in_channels, out_channels,   kernel_size=1, stride=1, padding=0)
        self.projector2=nn.Conv1d(skip_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.norm1=nn.GroupNorm(1, in_channels)
        self.norm2=nn.GroupNorm(1, skip_channels)

    def forward(self, skip, x):
        x = self.up(x)
        x = self.norm1(x)
        skip = self.norm2(skip)
        x = self.projector1(x)
        skip = self.projector2(skip)
        #print(x.shape, skip.shape)
        y = x+skip
        return y
#%%
class RegressionHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.head=nn.Linear(in_channels, out_channels)
        
    def forward(self, x):
        y=self.head(x)
        return y
#%%
class UNet(nn.Module):
    def __init__(self, h_dim):        
        super().__init__()
        self.h_dim=h_dim
        self.T0a=Block(in_channels=50*2*3, out_channels=h_dim, 
                       kernel_size=1, stride=1, padding=0)

        self.T1a=Block(in_channels=h_dim, out_channels=h_dim,
                       kernel_size=5, stride=5, padding=0)
                  
        self.T2a=Block(in_channels=h_dim, out_channels=h_dim,
                       kernel_size=5, stride=5, padding=0)        

        self.T3a=Block(in_channels=h_dim, out_channels=h_dim,
                       kernel_size=4, stride=4, padding=0)
        
        self.M3=MergeLayer(in_channels=h_dim, out_channels=h_dim, 
                           skip_channels=h_dim, scale_factor=4)
                  
        self.T2b=Block(in_channels=h_dim, out_channels=h_dim,
                       kernel_size=3, stride=1, padding=1)
        
        self.M2=MergeLayer(in_channels=h_dim, out_channels=h_dim, 
                           skip_channels=h_dim, scale_factor=5)        

        self.T1b=Block(in_channels=h_dim, out_channels=h_dim,
                       kernel_size=3, stride=1, padding=1)
        
        self.M1=MergeLayer(in_channels=h_dim, out_channels=h_dim, 
                           skip_channels=h_dim, scale_factor=5)        

        self.T0b=Block(in_channels=h_dim, out_channels=h_dim,
                       kernel_size=3, stride=1, padding=1)

        self.head=RegressionHead(h_dim, 50*3*2)
        
    
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
        x=x1
        
        x=self.get_patch(x)
        x=x.permute(0,2,1)
        
        #print('x', x.shape)
        t0a = self.T0a(x)
        #print("t0a.shape", t0a.shape)

        t1a = self.T1a(t0a)
        #print("t1a.shape", t1a.shape)

        t2a = self.T2a(t1a)
        #print("t2a.shape", t2a.shape)
        
        t3a = self.T3a(t2a)
        #print("t3a.shape", t3a.shape)
        
        t3b=t3a
        
        m3=self.M3(t2a, t3b)
        #print("m3.shape", m3.shape)
        
        t2b=self.T2b(m3)
        #print("t2b.shape", t2b.shape)
        
        m2=self.M2(t1a, t2b)
        #print("m2.shape", m2.shape)
        
        t1b=self.T1b(m2)
        #print("t1b.shape", t1b.shape)
        
        m1=self.M1(t0a, t1b)
        #print("m1.shape", m1.shape)
        
        t0b=self.T0b(m1)
        #print("t0b.shape", t0b.shape)
        
        t0b=t0b.permute(0,2,1) 
        #print("t0b.shape", t0b.shape)
        
        out=self.head(t0b)
        #print("out.shape", out.shape)
        #out (100, 50*3*2)
        out=self.un_patch(out)
        #print("out.shape", out.shape)
        
        return out
#%%
class UNet1(nn.Module):
    def __init__(self, h_dim):        
        super().__init__()
        self.h_dim=h_dim
        self.T0a=Block(in_channels=50*2*3, out_channels=h_dim//8, 
                       kernel_size=1, stride=1, padding=0)

        self.T1a=Block(in_channels=h_dim//8, out_channels=h_dim//4,
                       kernel_size=5, stride=5, padding=0)
                  
        self.T2a=Block(in_channels=h_dim//4, out_channels=h_dim,
                       kernel_size=5, stride=5, padding=0)        

        self.T3a=Block(in_channels=h_dim, out_channels=h_dim,
                       kernel_size=4, stride=4, padding=0)
        
        self.M3=MergeLayer(in_channels=h_dim, out_channels=h_dim, 
                           skip_channels=h_dim, scale_factor=4)
                  
        self.T2b=Block(in_channels=h_dim, out_channels=h_dim//4,
                       kernel_size=3, stride=1, padding=1)
        
        self.M2=MergeLayer(in_channels=h_dim//4, out_channels=h_dim//4, 
                           skip_channels=h_dim//4, scale_factor=5)        

        self.T1b=Block(in_channels=h_dim//4, out_channels=h_dim//8,
                       kernel_size=3, stride=1, padding=1)
        
        self.M1=MergeLayer(in_channels=h_dim//8, out_channels=h_dim//8, 
                           skip_channels=h_dim//8, scale_factor=5)        

        self.T0b=Block(in_channels=h_dim//8, out_channels=h_dim//8,
                       kernel_size=3, stride=1, padding=1)

        self.head=RegressionHead(h_dim//8, 50*3*2)
        
    
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
        x=x1
        
        x=self.get_patch(x)
        x=x.permute(0,2,1)
        
        #print('x', x.shape)
        t0a = self.T0a(x)
        #print("t0a.shape", t0a.shape)

        t1a = self.T1a(t0a)
        #print("t1a.shape", t1a.shape)

        t2a = self.T2a(t1a)
        #print("t2a.shape", t2a.shape)
        
        t3a = self.T3a(t2a)
        #print("t3a.shape", t3a.shape)
        
        t3b=t3a
        
        m3=self.M3(t2a, t3b)
        #print("m3.shape", m3.shape)
        
        t2b=self.T2b(m3)
        #print("t2b.shape", t2b.shape)
        
        m2=self.M2(t1a, t2b)
        #print("m2.shape", m2.shape)
        
        t1b=self.T1b(m2)
        #print("t1b.shape", t1b.shape)
        
        m1=self.M1(t0a, t1b)
        #print("m1.shape", m1.shape)
        
        t0b=self.T0b(m1)
        #print("t0b.shape", t0b.shape)
        
        t0b=t0b.permute(0,2,1) 
        #print("t0b.shape", t0b.shape)
        
        out=self.head(t0b)
        #print("out.shape", out.shape)
        #out (100, 50*3*2)
        out=self.un_patch(out)
        #print("out.shape", out.shape)
        
        return out
#%%
class UNet2(nn.Module):
    def __init__(self, h_dim):        
        super().__init__()
        self.h_dim=h_dim
        self.T0a=Block(in_channels=50*2*3, out_channels=h_dim//16, 
                       kernel_size=1, stride=1, padding=0)

        self.T1a=Block(in_channels=h_dim//16, out_channels=h_dim//4,
                       kernel_size=5, stride=5, padding=0)
                  
        self.T2a=Block(in_channels=h_dim//4, out_channels=h_dim,
                       kernel_size=5, stride=5, padding=0)        

        self.T3a=Block(in_channels=h_dim, out_channels=h_dim,
                       kernel_size=4, stride=4, padding=0)
        
        self.M3=MergeLayer(in_channels=h_dim, out_channels=h_dim, 
                           skip_channels=h_dim, scale_factor=4)
                  
        self.T2b=Block(in_channels=h_dim, out_channels=h_dim//4,
                       kernel_size=3, stride=1, padding=1)
        
        self.M2=MergeLayer(in_channels=h_dim//4, out_channels=h_dim//4, 
                           skip_channels=h_dim//4, scale_factor=5)        

        self.T1b=Block(in_channels=h_dim//4, out_channels=h_dim//16,
                       kernel_size=3, stride=1, padding=1)
        
        self.M1=MergeLayer(in_channels=h_dim//16, out_channels=h_dim//16, 
                           skip_channels=h_dim//16, scale_factor=5)        

        self.T0b=Block(in_channels=h_dim//16, out_channels=h_dim//16,
                       kernel_size=3, stride=1, padding=1)

        self.head=RegressionHead(h_dim//16, 50*3*2)
        
    
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
        x=x1
        
        x=self.get_patch(x)
        x=x.permute(0,2,1)
        
        #print('x', x.shape)
        t0a = self.T0a(x)
        #print("t0a.shape", t0a.shape)

        t1a = self.T1a(t0a)
        #print("t1a.shape", t1a.shape)

        t2a = self.T2a(t1a)
        #print("t2a.shape", t2a.shape)
        
        t3a = self.T3a(t2a)
        #print("t3a.shape", t3a.shape)
        
        t3b=t3a
        
        m3=self.M3(t2a, t3b)
        #print("m3.shape", m3.shape)
        
        t2b=self.T2b(m3)
        #print("t2b.shape", t2b.shape)
        
        m2=self.M2(t1a, t2b)
        #print("m2.shape", m2.shape)
        
        t1b=self.T1b(m2)
        #print("t1b.shape", t1b.shape)
        
        m1=self.M1(t0a, t1b)
        #print("m1.shape", m1.shape)
        
        t0b=self.T0b(m1)
        #print("t0b.shape", t0b.shape)
        
        t0b=t0b.permute(0,2,1) 
        #print("t0b.shape", t0b.shape)
        
        out=self.head(t0b)
        #print("out.shape", out.shape)
        #out (100, 50*3*2)
        out=self.un_patch(out)
        #print("out.shape", out.shape)
        
        return out
#%%
if __name__ == '__main__':
    
    x=torch.randn((10000,3)) 
    model=UNet2(512)
    out=model(x,None)
    
   
        
        