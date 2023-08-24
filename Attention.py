import torch
from torch import bmm
import numpy as np
#%%
def scaled_dot_product_attention(q, k, v, activation):
    #q.shape (N, A, B)
    #k.shape (N, L, B)
    #v.shape (N, L, M)
    #y.shape (N, A, M)
    #N is batch_size
    #A is sequence length of q
    #B is feature dimmention of q and k
    #M is feature dimmention of y
    #-------------------------------------------------
    k=k.permute(0,2,1) # (N, L, B) -> (N, B, L)
    qk=bmm(q, k) # (N, A, B) * (N, B, L) => (N, A, L)
    scale=1/np.sqrt(q.shape[2])
    w=activation(qk) # softmax(qk, dim=2)
    w=w*scale
    y=bmm(w, v) # (N, A, L) * (N, L, M) => (N, A, M)
    return y

class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, activation):
        super().__init__()
        if activation == "softmax":
            self.activation=torch.nn.Softmax(dim=2)
        elif activation == "sigmoid":
            self.activation=torch.nn.Sigmoid()
        else:
            raise ValueError("unknown activation:"+str(activation))

    def forward(self, q, k, v):
        return scaled_dot_product_attention(q, k, v, self.activation)
