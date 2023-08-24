# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 22:24:13 2021

@author: liang
"""
import torch
import torch_scatter
from torch_sparse import SparseTensor
import numpy as np
from torch.linalg import det
from PolyhedronMesh import PolyhedronMesh
#%%
class TetrahedronMesh(PolyhedronMesh):
    #4-node C3D4/TET4 mesh
    def __init__(self):
        super().__init__()
        self.mesh_type='polyhedron_tet4'

    def build_adj_node_link(self):
        adj_node_link=[]
        for m in range(0, len(self.element)):
            id0=int(self.element[m][0])
            id1=int(self.element[m][1])
            id2=int(self.element[m][2])
            id3=int(self.element[m][3])
            adj_node_link.append([id0, id1]); adj_node_link.append([id1, id0])
            adj_node_link.append([id0, id2]); adj_node_link.append([id2, id0])
            adj_node_link.append([id0, id3]); adj_node_link.append([id3, id0])
            adj_node_link.append([id1, id2]); adj_node_link.append([id2, id1])
            adj_node_link.append([id1, id3]); adj_node_link.append([id3, id1])
            adj_node_link.append([id2, id3]); adj_node_link.append([id3, id2])
        adj_node_link=torch.tensor(adj_node_link, dtype=torch.int64)
        adj_node_link=torch.unique(adj_node_link, dim=0, sorted=True)
        self.adj_node_link=adj_node_link

    def build_edge(self):
        edge=[]
        for m in range(0, len(self.element)):
            id0=self.element[m][0]
            id1=self.element[m][1]
            id2=self.element[m][2]
            id3=self.element[m][3]
            if id0 < id1:
                edge.append([id0, id1])
            else:
                edge.append([id1, id0])
            if id0 < id2:
                edge.append([id0, id2])
            else:
                edge.append([id2, id0])
            if id0 < id3:
                edge.append([id0, id3])
            else:
                edge.append([id3, id0])
            if id1 < id2:
                edge.append([id1, id2])
            else:
                edge.append([id2, id1])
            if id1 < id3:
                edge.append([id1, id3])
            else:
                edge.append([id3, id1])
            if id2 < id3:
                edge.append([id2, id3])
            else:
                edge.append([id3, id2])
        edge=torch.tensor(edge, dtype=torch.int64)
        edge=torch.unique(edge, dim=0, sorted=True)
        self.edge=edge

    def cal_element_volumn(self):
        #need C3D4 element
        pass

    def subdivide(self):
        #draw a 3D figure and code this...
        pass

    def get_sub_mesh(self, element_id_list):
        #element.shape (M,4)
        element_sub=self.element[element_id_list]
        node_idlist, element_out=torch.unique(element_sub.reshape(-1), return_inverse=True)
        node_out=self.node[node_idlist]
        element_out=element_out.view(-1,4)
        mesh_new=TetrahedronMesh()
        mesh_new.node=node_out
        mesh_new.element=element_out
        return mesh_new
#%%
if __name__ == "__main__":
    pass
