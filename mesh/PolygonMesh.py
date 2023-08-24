# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 22:24:13 2021

@author: liang
"""
import torch
import numpy as np
from copy import deepcopy
from Mesh import Mesh
#%%
class PolygonMesh(Mesh):
    def __init__(self):
        super().__init__('polygon')

    def build_adj_node_link(self):
        adj_node_link=[]
        for m in range(0, len(self.element)):
            elm=self.element[m]
            for k in range(0, len(elm)):
                if k < len(elm)-1:
                    a=int(elm[k]); b=int(elm[k+1])
                else:
                    a=int(elm[k]); b=int(elm[0])
                adj_node_link.append([a, b])
                adj_node_link.append([b, a])
        adj_node_link=torch.tensor(adj_node_link, dtype=torch.int64)
        adj_node_link=torch.unique(adj_node_link, dim=0, sorted=True)
        self.adj_node_link=adj_node_link

    def build_edge(self):
        edge=[]
        for m in range(0, len(self.element)):
            elm=self.element[m]
            for k in range(0, len(elm)):
                if k < len(elm)-1:
                    a=int(elm[k]); b=int(elm[k+1])
                else:
                    a=int(elm[k]); b=int(elm[0])
                if a < b:
                    edge.append([a, b])
                else:
                    edge.append([b, a])
        edge=torch.tensor(edge, dtype=torch.int64)
        edge=torch.unique(edge, dim=0, sorted=True)
        self.edge=edge

    def update_edge_length(self):
        self.edge_length=PolygonMesh.cal_edge_length(self.node, self.edge)

    @staticmethod
    def cal_edge_length(node, edge):
        x_j=node[edge[:,0]]
        x_i=node[edge[:,1]]
        edge_length=torch.norm(x_i-x_j, p=2, dim=1, keepdim=True)
        return edge_length

    def find_boundary_node(self):
        #return index list of nodes on boundary
        if self.edge is None:
            self.build_edge()
        if self.edge_to_element_table["adj2"] is None:
            self.build_edge_to_element_table(adj=2)
        boundary=[]
        for k in range(0, len(self.edge)):
            elm=self.edge_to_element_table["adj2"][k]
            if len(elm) <= 1:
                boundary.append(int(self.edge[k,0]))
                boundary.append(int(self.edge[k,1]))
        boundary=np.unique(boundary)
        return boundary

    def is_quad(self):
        if isinstance(self.element, torch.Tensor):
            if len(self.element[0]) == 4:
                #this is QuadMesh
                return True
            else:
                return False
        m_list=[]
        for m in range(0, len(self.element)):
            m_list.append(len(self.element[m]))
        m_min=min(m_list)
        m_max=max(m_list)
        if m_min == m_max == 4:
            return True
        else:
            return False

    def is_tri(self):
        if isinstance(self.element, torch.Tensor):
            if len(self.element[0]) == 3:
                #this is TriangleMesh
                return True
            else:
                return False
        m_list=[]
        for m in range(0, len(self.element)):
            m_list.append(len(self.element[m]))
        m_min=min(m_list)
        m_max=max(m_list)
        if m_min == m_max == 3:
            return True
        else:
            return False

    def is_quad_tri(self):
        if isinstance(self.element, torch.Tensor):
            if len(self.element[0]) == 3 or len(self.element[0]) == 4:
                #this is QuadTriangleMesh
                return True
            else:
                return False
        m_list=[]
        for m in range(0, len(self.element)):
            m_list.append(len(self.element[m]))
        m_min=min(m_list)
        m_max=max(m_list)
        if (m_min == m_max == 3) or (m_min == m_max == 4):
            return True
        else:
            return False

    def quad_to_tri(self):
        if isinstance(self.element, torch.Tensor):
            if len(self.element[0]) == 3:
                #this is TriangleMesh
                return
        element=self.element
        element_new=[]
        m_list=[]
        for m in range(0, len(element)):
            elm=element[m]
            if len(elm) == 4:
                # x3------x2
                # |       |
                # |       |
                # x0------x1
                # cut along x0-x2
                #-----------
                id0=int(elm[0])
                id1=int(elm[1])
                id2=int(elm[2])
                id3=int(elm[3])
                element_new.append([id0, id2, id3])
                element_new.append([id0, id1, id2])
                m_list.append(3)
            elif len(elm) == 3:
                id0=int(elm[0])
                id1=int(elm[1])
                id2=int(elm[2])
                element_new.append([id0, id1, id2])
                m_list.append(3)
            else:
                element_new.append(elm)
                m_list.append(len(elm))
        if min(m_list) == max(m_list):
            element_new=torch.tensor(element_new, dtype=torch.int64)
        self.element=element_new
#%%
if __name__ == "__main__":
    filename="D:/MLFEA/TAA/data/bav17_AortaModel_P0_best.pt"
    aorta=PolygonMesh()
    #aorta.load_from_vtk(filename, "float32")
    aorta.load_from_torch(filename)
    aorta.node_data={'node_data1':torch.rand((len(aorta.node), 6)),
                     'node_data2':torch.rand((len(aorta.node), 6))}
    aorta.element_data={'element_data1':torch.rand((len(aorta.element), 6)),
                       'element_data2':torch.rand((len(aorta.element), 6))}
    aorta.save_by_vtk("F:/MLFEA/TAA/test_poly.vtk")
    #%%
    from time import time
    t0=time()
    aorta.build_adj_node_link()
    #aorta.build_edge()
    t1=time()
    aorta.build_node_to_edge_table()
    t2=time()
    print(t1-t0, t2-t1)
    #%%
    t0=time()
    boundary=aorta.find_boundary_node()
    t1=time()
    print(t1-t0)
    #%%
    t0=time()
    aorta.build_node_to_node_table()
    t1=time()
    print(t1-t0)
    #%%
    t0=time()
    aorta.build_element_to_element_table(adj=2)
    t1=time()
    print(t1-t0)

