# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 22:24:13 2021

@author: liang
"""
import torch
import torch_scatter
from torch_sparse import SparseTensor
import numpy as np
import json
from torch.linalg import det
from Element_C3D8 import cal_dh_dr, get_integration_point_1i, get_integration_point_8i, interpolate
from PolyhedronMesh import PolyhedronMesh
#%%
class HexahedronMesh(PolyhedronMesh):
    #8-node C3D8 mesh
    def __init__(self):
        super().__init__()
        self.mesh_type='polyhedron_hex8'

    def build_adj_node_link(self):
        adj_node_link=[]
        for m in range(0, len(self.element)):
            id0=int(self.element[m][0])
            id1=int(self.element[m][1])
            id2=int(self.element[m][2])
            id3=int(self.element[m][3])
            id4=int(self.element[m][4])
            id5=int(self.element[m][5])
            id6=int(self.element[m][6])
            id7=int(self.element[m][7])
            adj_node_link.append([id0, id1]); adj_node_link.append([id1, id0])
            adj_node_link.append([id1, id2]); adj_node_link.append([id2, id1])
            adj_node_link.append([id2, id3]); adj_node_link.append([id3, id2])
            adj_node_link.append([id3, id0]); adj_node_link.append([id0, id3])
            adj_node_link.append([id4, id5]); adj_node_link.append([id5, id4])
            adj_node_link.append([id5, id6]); adj_node_link.append([id6, id5])
            adj_node_link.append([id6, id7]); adj_node_link.append([id7, id6])
            adj_node_link.append([id7, id4]); adj_node_link.append([id4, id7])
            adj_node_link.append([id0, id4]); adj_node_link.append([id4, id0])
            adj_node_link.append([id1, id5]); adj_node_link.append([id5, id1])
            adj_node_link.append([id2, id6]); adj_node_link.append([id6, id2])
            adj_node_link.append([id3, id7]); adj_node_link.append([id7, id3])
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
            id4=self.element[m][4]
            id5=self.element[m][5]
            id6=self.element[m][6]
            id7=self.element[m][7]
            if id0 < id1:
                edge.append([id0, id1])
            else:
                edge.append([id1, id0])
            if id1 < id2:
                edge.append([id1, id2])
            else:
                edge.append([id2, id1])
            if id2 < id3:
                edge.append([id2, id3])
            else:
                edge.append([id3, id2])
            if id3 < id0:
                edge.append([id3, id0])
            else:
                edge.append([id0, id3])
            if id4 < id5:
                edge.append([id4, id5])
            else:
                edge.append([id5, id4])
            if id5 < id6:
                edge.append([id5, id6])
            else:
                edge.append([id6, id5])
            if id6 < id7:
                edge.append([id6, id7])
            else:
                edge.append([id7, id6])
            if id7 < id4:
                edge.append([id7, id4])
            else:
                edge.append([id4, id7])
            if id0 < id4:
                edge.append([id0, id4])
            else:
                edge.append([id4, id0])
            if id1 < id5:
                edge.append([id1, id5])
            else:
                edge.append([id5, id1])
            if id2 < id6:
                edge.append([id2, id6])
            else:
                edge.append([id6, id2])
            if id3 < id7:
                edge.append([id3, id7])
            else:
                edge.append([id7, id3])
        edge=torch.tensor(edge, dtype=torch.int64)
        edge=torch.unique(edge, dim=0, sorted=True)
        self.edge=edge

    def cal_element_volumn(self):
        X=self.node[self.element]#shape (M,8,3)
        r1i=get_integration_point_1i(X.dtype, X.device)
        dX_dr=cal_dh_dr(r1i, X)
        volumn=8*det(dX_dr)
        return volumn

    def subdivide(self):
        #draw a 3D figure and code this...
        pass

    def get_sub_mesh(self, element_id_list):
        #element.shape (M,8)
        element_sub=self.element[element_id_list]
        node_idlist, element_out=torch.unique(element_sub.reshape(-1), return_inverse=True)
        node_out=self.node[node_idlist]
        element_out=element_out.view(-1,8)
        mesh_new=HexahedronMesh()
        mesh_new.node=node_out
        mesh_new.element=element_out
        return mesh_new
#%%
if __name__ == "__main__":
    pass
