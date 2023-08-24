import torch
from torch.nn import Linear, Sequential, ReLU, GroupNorm
from torch_scatter import scatter
#%%
class MLP(torch.nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, n_layers, layer_norm):
        super().__init__()
        self.in_dim=in_dim
        self.h_dim=h_dim
        self.out_dim=out_dim
        self.n_layers=n_layers
        self.layer_norm=layer_norm
        mlp=[]
        mlp.append(Linear(in_dim, h_dim))
        mlp.append(ReLU())
        for n in range(0, n_layers):
            mlp.append(Linear(h_dim, h_dim))
            mlp.append(ReLU())
        mlp.append(Linear(h_dim, out_dim))
        if layer_norm==True:
            mlp.append(GroupNorm(1, out_dim))
        self.mlp=Sequential(*mlp)

    def forward(self, x):
        y=self.mlp(x)
        return y
#%%
class EdgeConv(torch.nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, n_layers, layer_norm):
        super().__init__()
        self.in_dim=in_dim
        self.h_dim=h_dim
        self.out_dim=out_dim
        self.n_layers=n_layers
        self.layer_norm=layer_norm
        self.mlp=MLP(3*in_dim, h_dim, out_dim, n_layers, layer_norm)

    def forward(self, x, edge_index, e):
        # x has shape [N, in_dim]
        # edge_index has shape [2, E]
        # e has shape [E, in_dim]
        #--------- message --------------
        x_j = x[edge_index[0]]  # Source node features [E, in_dim]
        x_i = x[edge_index[1]]  # Target node features [E, in_dim]
        in_=torch.cat([x_i, x_j, e], dim=1)
        msg=self.mlp(in_) # shape [E, out_dim]
        #---------------------------------------
        e_new=e+msg # residual connection
        return e_new
#%%
class NodeConv(torch.nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, n_layers, layer_norm, aggr):
        super().__init__()
        self.in_dim=in_dim
        self.h_dim=h_dim
        self.out_dim=out_dim
        self.n_layers=n_layers
        self.layer_norm=layer_norm
        self.aggr=aggr
        self.mlp=MLP(2*in_dim, h_dim, out_dim, n_layers, layer_norm)

    def forward(self, x, edge_index, e):
        # x has shape [N, in_dim]
        # edge_index has shape [2, E]
        # e has shape [E, in_dim]
        #--------- message --------------
        #msg_e shape [N, in_dim]
        msg_e = scatter(e, edge_index[1], dim=0, dim_size=x.shape[0], reduce=self.aggr)
        in_=torch.cat([x, msg_e], dim=1)
        msg=self.mlp(in_) # shape [N, in_dim]
        #---------------------------------------
        x_new=x+msg # residual connection
        return x_new
#%%
class GraphConv(torch.nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, n_layers, layer_norm, aggr):
        super().__init__()
        self.in_dim=in_dim
        self.h_dim=h_dim
        self.out_dim=out_dim
        self.n_layers=n_layers
        self.layer_norm=layer_norm
        self.aggr=aggr
        self.edge_conv=EdgeConv(in_dim, h_dim, out_dim, n_layers, layer_norm)
        self.node_conv=NodeConv(in_dim, h_dim, out_dim, n_layers, layer_norm, aggr)

    def forward(self, x, edge_index, e):
        e_new=self.edge_conv(x, edge_index, e)
        x_new=self.node_conv(x, edge_index, e_new)
        return x_new, e_new
#%%
class Processor(torch.nn.Module):
    def __init__(self, n_blocks, in_dim, h_dim, out_dim, n_layers, layer_norm, aggr):
        super().__init__()
        self.in_dim=in_dim
        self.h_dim=h_dim
        self.out_dim=out_dim
        self.n_layers=n_layers
        self.layer_norm=layer_norm
        self.aggr=aggr
        self.n_blocks=n_blocks
        self.block=torch.nn.ModuleList()
        for n in range(0, n_blocks):
            self.block.append(GraphConv(in_dim, h_dim, out_dim, n_layers, layer_norm, aggr))

    def forward(self, x, edge_index, e):
        for n in range(0, self.n_blocks):
            x, e=self.block[n](x, edge_index, e)
            #print(x.shape, e.shape)
        return x, e
#%%
class MeshGraphNet(torch.nn.Module):
    def __init__(self, x_dim, e_dim, n_blocks, h_dim, out_dim, x_scale=1, n_layers=1, aggr='mean'):
        super().__init__()
        self.x_dim=x_dim
        self.e_dim=e_dim
        self.h_dim=h_dim
        self.out_dim=out_dim
        self.x_scale=x_scale
        self.n_layers=n_layers
        self.aggr=aggr
        self.encoder_x=MLP(x_dim, h_dim, h_dim, n_layers, True)
        self.encoder_e=MLP(e_dim, h_dim, h_dim, n_layers, True)
        self.processor=Processor(n_blocks, h_dim, h_dim, h_dim, n_layers, True, aggr)
        self.decoder=MLP(h_dim, h_dim, out_dim, n_layers, False)

    def forward(self, x, edge_index, e, x_ref=None):
        if self.x_scale != 1:
            x=x*self.x_scale
        x=self.encoder_x(x)
        e=self.encoder_e(e)
        x, e=self.processor(x, edge_index, e)
        y=self.decoder(x)
        return y
#%%
from NNFEA_net_x_c import BaseNet0,  Net1
#--------------------------------------------
class Encoder3(torch.nn.Module):
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
                       +'x_dim=3'
                       +',h_dim='+str(h_dim)
                       +',n_layers='+str(n_layers)
                       +',x_scale='+str(x_scale)
                       +',alpha='+str(alpha)
                       +',gamma='+str(gamma)
                       +',y_dim='+str(x_dim*y_dim)
                       +")"
                       )
    def forward(self, x, x_ref):
        #x.shape (N, x_dim)
        #x_ref.shape (N, 3)
        y1=self.netA(x_ref)
        #y.shape (N, y_dim*x1_dim)
        y1=y1.view(x.shape[0], self.x_dim, self.y_dim)
        x1=x.view(x.shape[0], self.x_dim, 1)
        y=(y1*x1).mean(dim=(0,1))
        y=y.view(1,-1)
        y=y.expand(x1.shape[0], self.y_dim)
        return y
#--------------------------------------------
class MeshGraphNet1(torch.nn.Module):
    def __init__(self, x_dim, e_dim, n_blocks, h_dim, out_dim, x_scale=1, n_layers=1, aggr='mean'):
        super().__init__()
        self.x_dim=x_dim
        self.e_dim=e_dim
        self.h_dim=h_dim
        self.out_dim=out_dim
        self.x_scale=x_scale
        self.n_layers=n_layers
        self.aggr=aggr
        self.encoder_x=MLP(x_dim, h_dim, h_dim, n_layers, True)
        self.encoder_e=MLP(e_dim, h_dim, h_dim, n_layers, True)
        self.processor=Processor(n_blocks, h_dim, h_dim, h_dim, n_layers, True, aggr)

        if out_dim == 3: #shape
            self.encoder=eval("Encoder3('BaseNet0',h_dim,h_dim,2,1,1,1,3)")
            self.decoder=eval("Net1('BaseNet5b',3,3,512,4,1,1,1,3,'sigmoid')")
        elif out_dim == 9: #stress
            self.encoder=eval("Encoder3('BaseNet0',h_dim,h_dim,2,1,1,1,3)")
            self.decoder=eval("Net1('BaseNet5b',3,3,512,4,1,1,1,9,'sigmoid')")

    def forward(self, x, edge_index, e, x_ref):
        if self.x_scale != 1:
            x=x*self.x_scale
        x=self.encoder_x(x)
        e=self.encoder_e(e)
        x, e=self.processor(x, edge_index, e)
        c=self.encoder(x, x_ref)
        y=self.decoder(x_ref, c)
        return y
#%%
class MeshGraphNet2(torch.nn.Module):
    def __init__(self, x_dim, e_dim, n_blocks, h_dim, out_dim, x_scale=1, n_layers=1, aggr='mean'):
        super().__init__()
        self.x_dim=x_dim
        self.e_dim=e_dim
        self.h_dim=h_dim
        self.out_dim=out_dim
        self.x_scale=x_scale
        self.n_layers=n_layers
        self.aggr=aggr
        self.encoder_x=MLP(x_dim, h_dim, h_dim, n_layers, True)
        self.encoder_e=MLP(e_dim, h_dim, h_dim, n_layers, True)
        self.encoder_c=MLP(h_dim, h_dim, h_dim, n_layers, True)
        self.processor=Processor(n_blocks, h_dim, h_dim, h_dim, n_layers, True, aggr)
        self.decoder=MLP(2*h_dim, h_dim, out_dim, n_layers, False)

    def forward(self, x, edge_index, e, x_ref=None):
        if self.x_scale != 1:
            x=x*self.x_scale
        x=self.encoder_x(x)
        e=self.encoder_e(e)
        x, e=self.processor(x, edge_index, e)
        c=self.encoder_c(x.mean(dim=0, keepdim=True))
        c=c.expand(x.shape[0], self.h_dim)
        xc=torch.cat([x,c],dim=1)
        y=self.decoder(xc)
        return y
#%%
if __name__ == '__main__':
    #%%
    import sys
    sys.path.append("../../../MLFEA/code/mesh")
    import numpy as np
    from IPython import display
    import matplotlib.pyplot as plt
    import torch
    from QuadMesh import QuadMesh
    from PolyhedronMesh import PolyhedronMesh

    mesh_px=PolyhedronMesh()
    mesh_px.load_from_torch("F:/MLFEA/TAA/data/343c1.5/p0_0_solid.pt")
    mesh_px.build_adj_node_link()
    #%%
    edge_index=mesh_px.adj_node_link.t()
    x=mesh_px.node.to(torch.float32)
    e=torch.rand((edge_index.shape[1], 10), dtype=torch.float32)
    #%%
    net=MeshGraphNet(x_dim=3, e_dim=10, n_blocks=2, h_dim=128, out_dim=3)
    y=net(x, edge_index, e)
    #%%
    net=MeshGraphNet1(x_dim=3, e_dim=10, n_blocks=2, h_dim=128, out_dim=3)
    y=net(x, edge_index, e, x)
    #%%
    net=MeshGraphNet(x_dim=3, e_dim=10, n_blocks=2, h_dim=128, out_dim=3)
    y=net(x, edge_index, e)



